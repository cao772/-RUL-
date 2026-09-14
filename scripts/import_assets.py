from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "checkpoints"

ALLOW = {
    "dkasc84_pretrain/features_mean.npy": "dkasc84_pretrain/features_mean.npy",
    "dkasc84_pretrain/features_std.npy": "dkasc84_pretrain/features_std.npy",
    "peft_1403/features_mean.npy": "peft_1403/features_mean.npy",
    "peft_1403/features_std.npy": "peft_1403/features_std.npy",
    "peft_1403/target_stats.npy": "peft_1403/target_stats.npy",
    "peft_1403/rate_stats.npy": "peft_1403/rate_stats.npy",
    "peft_1403/pv_rul_mamba_peft.pth": "peft_1403/pv_rul_mamba_peft.pth",
    "peft_1403/X_windows.npy": "peft_1403/X_windows.npy",
    "peft_1403/y_rul.npy": "peft_1403/y_rul.npy",
    "peft_1403/y_rate.npy": "peft_1403/y_rate.npy",
    "peft_1403/validation_report.json": "peft_1403/validation_report.json",
}


def norm(name: str) -> str:
    return name.replace("\\", "/")


def find_member(names: list[str], suffix: str) -> str | None:
    target = "/T6/checkpoints/" + suffix
    for name in names:
        n = "/" + norm(name).lstrip("/")
        if n.endswith(target):
            return name
    return None


def import_from_zip(zip_path: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到 ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        missing = [suffix for suffix in ALLOW if find_member(names, suffix) is None]
        if missing:
            raise RuntimeError("ZIP 中缺少 V1 必需资产:\n- " + "\n- ".join(missing))
        for suffix, dest_rel in ALLOW.items():
            member = find_member(names, suffix)
            assert member is not None
            dest = DEST / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"[OK] {dest_rel}  ({dest.stat().st_size / 1024:.1f} KB)")

    print("\n资产导入完成。海南 data/hainan_* 与 hainan_results/* 未导入。")
    print("下一步: python scripts/smoke_test.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="从客户提供的光伏组件RUL预测.zip中导入 V1 非海南运行资产")
    parser.add_argument("zip_path", type=Path, help="客户 ZIP 文件路径")
    args = parser.parse_args()
    try:
        import_from_zip(args.zip_path.expanduser().resolve())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
