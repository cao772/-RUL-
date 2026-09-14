from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.inference import PredictionEngine

if __name__ == "__main__":
    e = PredictionEngine(ROOT)
    for idx in (0, 100, e.n_samples - 1):
        r = e.predict_sample(idx)
        print(idx, r["rul_days"], r["rate_pct_per_year"], r["forecast_end_pmax_ratio"])
    print("SMOKE_OK")
