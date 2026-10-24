"""Synthetic view-identity/RNG checks; no model fitting."""
import random
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from train_views import epoch_features

english=[object() for _ in range(5)]
spanish=[object() for _ in range(5)]
pairs=["pairA","pairA","pairB","pairB","pairC"]
eligible=[True,True,False,False,True]
before=random.getstate()
control,selected=epoch_features(english,spanish,pairs,eligible,random.Random(10),0)
assert all(a is b for a,b in zip(english,control)) and not any(selected)
all_views,selected=epoch_features(english,spanish,pairs,eligible,random.Random(10),1)
assert selected==eligible
assert all_views[0] is spanish[0] and all_views[2] is english[2]
for seed in range(10):
    _,selected=epoch_features(english,spanish,pairs,eligible,random.Random(seed),.3)
    assert selected[0]==selected[1] and selected[2]==selected[3]
assert before==random.getstate(),"View sampling must not change global RNG"
print("VIEW_CHECKS probability0 exact feature identity, excluded-pair fallback, paired choices, global RNG isolation passed")
