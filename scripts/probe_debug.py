"""Sanity check: generate two prompts with SD-Turbo and SDXL-Turbo, compare images."""
import hashlib
from pathlib import Path

import pandas as pd
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "debug"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "configs" / "probe_sample.csv").head(2)
print(df[["probe_id", "text1", "text2", "label"]].to_string(index=False), flush=True)

for gname, mid in (("sd-turbo", "stabilityai/sd-turbo"), ("sdxl-turbo", "stabilityai/sdxl-turbo")):
    print(f"\n=== {gname} ===", flush=True)
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(mid, torch_dtype=torch.float16, safety_checker=None)
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(mid, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for i, r in df.iterrows():
        for side, text in (("1", r.text1), ("2", r.text2)):
            gen = torch.Generator(device="cuda").manual_seed(1000 + i)
            im = pipe(prompt=str(text), num_inference_steps=1, guidance_scale=0.0, generator=gen).images[0]
            p = OUT / f"{gname}_{r.probe_id}_{side}.png"
            im.save(p)
            print(f"{p.name} md5={hashlib.md5(p.read_bytes()).hexdigest()} size={im.size}", flush=True)
    del pipe
    torch.cuda.empty_cache()

print("\n=== directory diff ===", flush=True)
for p in sorted(OUT.glob("sd-turbo_*.png")):
    q = OUT / p.name.replace("sd-turbo_", "sdxl-turbo_")
    if q.exists():
        h1 = hashlib.md5(p.read_bytes()).hexdigest()
        h2 = hashlib.md5(q.read_bytes()).hexdigest()
        print(f"{p.name}: same={h1 == h2} {h1[:8]} {h2[:8]}", flush=True)

print("\n=== CLIP-B32 similarity ===", flush=True)
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to("cuda").eval()
with torch.no_grad():
    for i, r in df.iterrows():
        for gname in ("sd-turbo", "sdxl-turbo"):
            p1 = OUT / f"{gname}_{r.probe_id}_1.png"
            p2 = OUT / f"{gname}_{r.probe_id}_2.png"
            inp = proc(images=[Image.open(p1).convert("RGB"), Image.open(p2).convert("RGB")],
                       return_tensors="pt").to("cuda")
            f = model.get_image_features(**inp).float()
            f = f / f.norm(dim=-1, keepdim=True)
            print(f"{gname} {r.probe_id}: cos={(f[0] @ f[1]).item():.6f}", flush=True)
