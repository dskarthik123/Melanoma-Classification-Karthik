import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import make_loaders

ROOT = Path("/user/HS402/kd01071/Downloads/Melanoma-Classification-Group26")

tr, va, meta_dim = make_loaders(
    csv=ROOT / "train.csv", split_csv="splits/patient.csv",
    img_dir=ROOT / "train", fold_col="patient_fold", balanced=True)

print(f"metadata dim: {meta_dim}")
print(f"train batches: {len(tr)}  val batches: {len(va)}")

t0 = time.time()
img, meta, y = next(iter(tr))
print(f"first batch in {time.time() - t0:.1f}s")
print(f"image {tuple(img.shape)}  meta {tuple(meta.shape)}  target {tuple(y.shape)}")
print(f"positive fraction in balanced batch: {y.mean():.2f}")