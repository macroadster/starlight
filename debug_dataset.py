# debug_dataset.py
from stego_dataset_v2 import StegoPairDataset
import albumentations as A

train_tf = A.Compose([A.ToFloat(max_value=255)], additional_targets={'mask':'image'})

ds = StegoPairDataset(
    clean_root="datasets/sample_submission_2025/clean",
    stego_root="datasets/sample_submission_2025/stego",
    msg_len=100,
    transform=train_tf,
    strict=False,          # prints warnings instead of crashing
)

print("Pairs found:", len(ds))
for i, p in enumerate(ds.pairs[:5]):
    print(f"  [{i}] clean={p['clean']}  stego={p['stego']}  json={p['json']}")
