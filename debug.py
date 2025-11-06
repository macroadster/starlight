# debug.py
from stego_dataset_v2 import StegoPairDataset
import albumentations as A

tf = A.Compose([], additional_targets={'mask':'image'})

ds = StegoPairDataset(
    clean_root="datasets/sample_submission_2025/clean",
    stego_root="datasets/sample_submission_2025/stego",
    msg_len=100,
    transform=tf,
    strict=False
)

print(f"Total pairs: {len(ds)}")
