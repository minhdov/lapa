
from pathlib import Path
from collections import Counter

DA_ROOT = Path("/datasets/ssv2_libero90/libero-10_images_ssv2")
GT_ROOT = Path("/datasets/ssv2_libero90/libero-10_depth_ssv2_da")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(folder: Path):
    if not folder.exists():
        return None
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

da_folders = {p.name: p for p in DA_ROOT.iterdir() if p.is_dir()}
gt_folders = {p.name: p for p in GT_ROOT.iterdir() if p.is_dir()}

all_names = sorted(set(da_folders) | set(gt_folders))

same = []
diff = []
missing_in_da = []
missing_in_gt = []

for name in all_names:
    da_path = da_folders.get(name)
    gt_path = gt_folders.get(name)

    if da_path is None:
        missing_in_da.append(name)
        continue

    if gt_path is None:
        missing_in_gt.append(name)
        continue

    da_n = count_images(da_path)
    gt_n = count_images(gt_path)

    if da_n == gt_n:
        same.append((name, da_n, gt_n))
    else:
        diff.append((name, da_n, gt_n, da_n - gt_n))

print("========== SUMMARY ==========")
print("DA root:", DA_ROOT)
print("GT root:", GT_ROOT)
print("DA folders:", len(da_folders))
print("GT folders:", len(gt_folders))
print("Common folders:", len(set(da_folders) & set(gt_folders)))
print("Same frame count:", len(same))
print("Different frame count:", len(diff))
print("Missing in DA:", len(missing_in_da))
print("Missing in GT:", len(missing_in_gt))

print("\n========== FRAME TOTALS ON COMMON FOLDERS ==========")
print("DA total frames:", sum(x[1] for x in same) + sum(x[1] for x in diff))
print("GT total frames:", sum(x[2] for x in same) + sum(x[2] for x in diff))
print("Total diff DA-GT:", sum(x[3] for x in diff))

print("\n========== TOP 30 DIFFERENT FOLDERS ==========")
for name, da_n, gt_n, delta in sorted(diff, key=lambda x: abs(x[3]), reverse=True)[:30]:
    print(f"{name}: DA={da_n}, GT={gt_n}, diff={delta}")

print("\n========== DIFF DISTRIBUTION ==========")
counter = Counter(delta for _, _, _, delta in diff)
for delta, c in counter.most_common(20):
    print(f"diff={delta}: folders={c}")
