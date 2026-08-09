import json, argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEED = 42
META = ["age_approx", "sex", "anatom_site_general_challenge"]


def audit(df: pd.DataFrame, img_dir: Path) -> dict:
    """Verify integrity and summarise structure before any modelling."""
    per_patient = df.groupby("patient_id")["target"].agg(["size", "sum"])
    missing = [n for n in df.image_name if not (img_dir / f"{n}.jpg").exists()]
    return {
        "n_images": len(df),
        "n_patients": df.patient_id.nunique(),
        "duplicate_image_names": int(df.image_name.duplicated().sum()),
        "missing_image_files": len(missing),
        "n_positive": int(df.target.sum()),
        "positive_rate": float(df.target.mean()),
        "imbalance_ratio": float((len(df) - df.target.sum()) / df.target.sum()),
        "images_per_patient": {
            "mean": float(per_patient["size"].mean()),
            "median": float(per_patient["size"].median()),
            "max": int(per_patient["size"].max()),
        },
        "patients_with_melanoma": int((per_patient["sum"] > 0).sum()),
        "metadata_missing": {c: int(df[c].isna().sum()) for c in META},
    }


def patient_split(df: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    """Assign fold 0 as validation; no patient_id crosses the boundary."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(df, df.target, groups=df.patient_id))
    s = pd.Series("train", index=df.index)
    s.iloc[va] = "val"
    return s


def image_split(df: pd.DataFrame) -> pd.Series:
    """Deliberately leaky control: stratified on label, ignoring patient_id."""
    tr, va = train_test_split(
        df.index, test_size=0.2, stratify=df.target, random_state=SEED
    )
    s = pd.Series("train", index=df.index)
    s.loc[va] = "val"
    return s


def verify(df: pd.DataFrame, col: str) -> dict:
    """Quantify patient overlap — the core claim of this project must be checked."""
    tr = set(df.loc[df[col] == "train", "patient_id"])
    va = set(df.loc[df[col] == "val", "patient_id"])
    val = df[df[col] == "val"]
    return {
        "shared_patients": len(tr & va),
        "val_images": len(val),
        "val_positives": int(val.target.sum()),
        "val_positive_rate": float(val.target.mean()),
    }


def main(csv: Path, img_dir: Path, out: Path):
    df = pd.read_csv(csv).reset_index(drop=True)
    report = {"audit": audit(df, img_dir)}

    df["patient_fold"] = patient_split(df)
    df["image_fold"] = image_split(df)
    report["patient_split"] = verify(df, "patient_fold")
    report["image_split"] = verify(df, "image_fold")

    assert report["patient_split"]["shared_patients"] == 0, "patient leakage detected"

    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    df[["image_name", "patient_id", "target", "patient_fold"]].to_csv(
        out / "splits/patient.csv", index=False)
    df[["image_name", "patient_id", "target", "image_fold"]].to_csv(
        out / "splits/image.csv", index=False)
    (out / "results/audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("."))
    main(*vars(p.parse_args()).values())