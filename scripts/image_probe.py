"""Caption-to-image + image-embedding probe for EQ vs NO.

No training. For each sampled row:
  - generate one image per caption with a text-to-image model;
  - embed both images with an image encoder;
  - cosine similarity is the EQ score.
A text-text cosine from the same encoder is the baseline. We report AUC and
best-threshold accuracy for each (generator, embedder, split).
"""
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "configs" / "probe_sample.csv"
OUT = ROOT / "outputs"
IMG = OUT / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE} torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0), flush=True)

GENERATORS = {
    "sd-turbo": {"id": "stabilityai/sd-turbo", "steps": 1, "guidance": 0.0},
    "sdxl-turbo": {"id": "stabilityai/sdxl-turbo", "steps": 1, "guidance": 0.0},
}
EMBEDDERS = {
    "clip-b32": {"id": "openai/clip-vit-base-patch32", "kind": "clip"},
    "clip-l14": {"id": "openai/clip-vit-large-patch14", "kind": "clip"},
    "siglip": {"id": "google/siglip-base-patch16-224", "kind": "siglip"},
    "dinov2": {"id": "facebook/dinov2-base", "kind": "dinov2"},
}


def generate_images(df):
    from diffusers import AutoPipelineForText2Image
    for gname, g in GENERATORS.items():
        print(f"\n=== GENERATOR {gname} ({g['id']}) ===", flush=True)
        pipe = None
        try:
            kwargs = {"torch_dtype": torch.float16}
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(g["id"], variant="fp16", **kwargs)
            except Exception:
                pipe = AutoPipelineForText2Image.from_pretrained(g["id"], **kwargs)
            pipe = pipe.to(DEVICE)
            pipe.set_progress_bar_config(disable=True)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            for i, r in df.iterrows():
                for side, text in (("1", r.text1), ("2", r.text2)):
                    path = IMG / gname / f"{r.probe_id}_{side}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        continue
                    try:
                        gen = torch.Generator(device=DEVICE).manual_seed(1000 + i)
                        im = pipe(prompt=str(text), num_inference_steps=g["steps"],
                                  guidance_scale=g["guidance"], generator=gen).images[0]
                        im.save(path)
                    except Exception as e:
                        print(f"GEN_ROW_FAIL {gname} {r.probe_id} {side}: {type(e).__name__}: {e}", flush=True)
                if (i + 1) % 20 == 0:
                    print(f"  generated {i + 1}/{len(df)}", flush=True)
        except Exception as e:
            print(f"GEN_FAIL {gname}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            try:
                del pipe
            except Exception:
                pass
            torch.cuda.empty_cache()


def load_embedder(name):
    if name == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(EMBEDDERS[name]["id"])
        model = AutoModel.from_pretrained(EMBEDDERS[name]["id"], torch_dtype=torch.float16).to(DEVICE).eval()
        return proc, model, "dinov2"
    if EMBEDDERS[name]["kind"] == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        proc = SiglipProcessor.from_pretrained(EMBEDDERS[name]["id"])
        model = SiglipModel.from_pretrained(EMBEDDERS[name]["id"], torch_dtype=torch.float16).to(DEVICE).eval()
        return proc, model, "siglip"
    from transformers import CLIPModel, CLIPProcessor
    proc = CLIPProcessor.from_pretrained(EMBEDDERS[name]["id"])
    model = CLIPModel.from_pretrained(EMBEDDERS[name]["id"], torch_dtype=torch.float16).to(DEVICE).eval()
    return proc, model, "clip"


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def image_sim(proc, model, kind, p1, p2):
    im1, im2 = Image.open(p1).convert("RGB"), Image.open(p2).convert("RGB")
    with torch.no_grad():
        if kind == "dinov2":
            inp = proc(images=[im1, im2], return_tensors="pt").to(DEVICE)
            out = model(**inp)
            f = unit(out.last_hidden_state[:, 0].float())
        elif kind == "siglip":
            inp = proc(images=[im1, im2], return_tensors="pt").to(DEVICE)
            f = unit(model.get_image_features(**inp).float())
        else:
            inp = proc(images=[im1, im2], return_tensors="pt").to(DEVICE)
            f = unit(model.get_image_features(**inp).float())
    return float((f[0] @ f[1]).item())


def text_sim(proc, model, kind, t1, t2):
    if kind == "dinov2":
        return np.nan
    with torch.no_grad():
        if kind == "siglip":
            inp = proc(text=[str(t1), str(t2)], return_tensors="pt", padding="max_length",
                       truncation=True, max_length=64).to(DEVICE)
            f = unit(model.get_text_features(**inp).float())
        else:
            inp = proc(text=[str(t1), str(t2)], return_tensors="pt", padding=True,
                       truncation=True, max_length=77).to(DEVICE)
            f = unit(model.get_text_features(**inp).float())
    return float((f[0] @ f[1]).item())


def evaluate(df):
    from sklearn.metrics import roc_auc_score

    scores = []
    for gname in GENERATORS:
        for ename in EMBEDDERS:
            proc = model = None
            try:
                proc, model, kind = load_embedder(ename)
            except Exception as e:
                print(f"EMB_LOAD_FAIL {gname} {ename}: {type(e).__name__}: {e}", flush=True)
                continue
            try:
                for _, r in df.iterrows():
                    p1 = IMG / gname / f"{r.probe_id}_1.png"
                    p2 = IMG / gname / f"{r.probe_id}_2.png"
                    if not (p1.exists() and p2.exists()):
                        continue
                    try:
                        isim = image_sim(proc, model, kind, p1, p2)
                    except Exception as e:
                        print(f"IMG_EMB_FAIL {gname} {ename} {r.probe_id}: {type(e).__name__}: {e}", flush=True)
                        isim = np.nan
                    try:
                        tsim = text_sim(proc, model, kind, r.text1, r.text2)
                    except Exception as e:
                        tsim = np.nan
                    scores.append(dict(generator=gname, embedder=ename, probe_id=r.probe_id,
                                       split=r.split, label=r.label, is_hard=pd.notna(r.category),
                                       image_sim=isim, text_sim=tsim))
            finally:
                try:
                    del model, proc
                except Exception:
                    pass
                torch.cuda.empty_cache()
    s = pd.DataFrame(scores)
    s.to_csv(OUT / "image-probe-scores.csv", index=False)
    print(f"\nscores rows={len(s)}", flush=True)
    return s


def summary(s):
    out = []
    for (g, e), d in s.groupby(["generator", "embedder"]):
        for split in ["all", "train", "dev", "trial", "hard"]:
            if split == "all":
                x = d
            elif split == "hard":
                x = d[d.is_hard]
            else:
                x = d[d.split == split]
            x = x.dropna(subset=["image_sim"])
            if len(x) < 6 or x.label.nunique() < 2:
                continue
            y = (x.label == "EQUIVALENCE").astype(int).values
            for metric in ["image_sim", "text_sim"]:
                vals = x[metric].values
                ok = ~np.isnan(vals)
                if ok.sum() < 6 or len(set(y[ok])) < 2:
                    continue
                auc = roc_auc_score(y[ok], vals[ok])
                # best-threshold accuracy
                best = 0.0
                for t in np.unique(vals[ok]):
                    acc = ((vals[ok] >= t).astype(int) == y[ok]).mean()
                    best = max(best, float(acc))
                eq = vals[ok][y[ok] == 1].mean()
                no = vals[ok][y[ok] == 0].mean()
                out.append(dict(generator=g, embedder=e, split=split, metric=metric, n=int(ok.sum()),
                                auc=float(auc), best_acc=best, mean_eq=float(eq), mean_no=float(no),
                                gap=float(eq - no)))
    res = pd.DataFrame(out)
    res.to_csv(OUT / "image-probe-results.csv", index=False)
    return res


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE)
    print(f"sample rows={len(df)}", flush=True)
    print(df.groupby(["split", "label"]).size().to_string(), flush=True)
    generate_images(df)
    s = evaluate(df)
    res = summary(s)
    if len(res):
        piv = res[res.metric == "image_sim"].sort_values(["split", "auc"], ascending=[True, False])
        print("\n=== IMAGE SIMILARITY (AUC, best acc) ===", flush=True)
        print(piv[["generator", "embedder", "split", "n", "auc", "best_acc", "mean_eq", "mean_no", "gap"]].to_string(index=False), flush=True)
        txt = res[res.metric == "text_sim"].sort_values(["split", "auc"], ascending=[True, False])
        print("\n=== TEXT BASELINE (AUC, best acc) ===", flush=True)
        print(txt[["generator", "embedder", "split", "n", "auc", "best_acc", "gap"]].to_string(index=False), flush=True)
    else:
        print("NO RESULTS", flush=True)
        sys.exit(1)
