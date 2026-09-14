from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.model.model_common import PV_RUL_Mamba, HIDDEN_DIM, NUM_LAYERS, DROPOUT

FEATURE_NAMES = [
    "power_ratio",
    "irradiance",
    "ambient_temp",
    "pmax_ratio",
    "module_temp",
    "rh",
]
FEATURE_LABELS = {
    "power_ratio": "功率比",
    "irradiance": "辐照度",
    "ambient_temp": "环境温度",
    "pmax_ratio": "Pmax健康指数",
    "module_temp": "组件温度",
    "rh": "相对湿度",
}
FEATURE_UNITS = {
    "power_ratio": "p.u.",
    "irradiance": "W/m²",
    "ambient_temp": "℃",
    "pmax_ratio": "p.u.",
    "module_temp": "℃",
    "rh": "%",
}
WINDOW_SIZE = 60
RUL_MIN_DAYS = 30.0
RUL_MAX_DAYS = 10950.0


@dataclass
class PredictionEngine:
    root: Path

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_root = self.root / "assets" / "checkpoints"
        self.pretrain_dir = self.ckpt_root / "dkasc84_pretrain"
        self.peft_dir = self.ckpt_root / "peft_1403"

        self.X = np.load(self.peft_dir / "X_windows.npy", mmap_mode="r")
        self.y_rul = np.load(self.peft_dir / "y_rul.npy", mmap_mode="r")
        self.y_rate = np.load(self.peft_dir / "y_rate.npy", mmap_mode="r")
        self.validation = json.loads((self.peft_dir / "validation_report.json").read_text(encoding="utf-8"))

        mean5 = np.load(self.pretrain_dir / "features_mean.npy").astype(np.float32)
        std5 = np.load(self.pretrain_dir / "features_std.npy").astype(np.float32)
        mean6 = np.load(self.peft_dir / "features_mean.npy").astype(np.float32)
        std6 = np.load(self.peft_dir / "features_std.npy").astype(np.float32)
        self.feat_mean = np.concatenate([mean5, [mean6[5]]]).astype(np.float32)
        self.feat_std = np.concatenate([std5, [std6[5]]]).astype(np.float32)

        self.y_mean, self.y_std = [float(v) for v in np.load(self.peft_dir / "target_stats.npy")]
        self.r_mean, self.r_std = [float(v) for v in np.load(self.peft_dir / "rate_stats.npy")]

        flat = np.asarray(self.X[:, :, 0, :], dtype=np.float32).reshape(-1, 6)
        p2 = np.percentile(flat, 2, axis=0)
        p98 = np.percentile(flat, 98, axis=0)
        self.clip_lo = ((p2 - self.feat_mean) / self.feat_std).astype(np.float32)
        self.clip_hi = ((p98 - self.feat_mean) / self.feat_std).astype(np.float32)

        self.model = PV_RUL_Mamba(
            input_dim=6,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(self.device)
        state = torch.load(
            self.peft_dir / "pv_rul_mamba_peft.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    def sample_raw(self, sample_index: int) -> np.ndarray:
        if not 0 <= sample_index < self.n_samples:
            raise ValueError(f"sample_index 必须在 0~{self.n_samples - 1} 之间")
        # 客户快照中的两个通道是同一序列的重复副本，业务演示固定使用第 1 路。
        return np.asarray(self.X[sample_index, :, 0, :], dtype=np.float32)

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.shape != (WINDOW_SIZE, 6):
            raise ValueError(f"模型输入必须是 {WINDOW_SIZE}×6，当前为 {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("模型输入存在 NaN 或 Infinity")
        if np.any(arr[:, 1] < 0):
            raise ValueError("辐照度不能为负数")
        if np.any((arr[:, 5] < 0) | (arr[:, 5] > 100)):
            raise ValueError("相对湿度必须在 0~100 之间")
        return arr

    @staticmethod
    def _rows(raw: np.ndarray) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for day_idx, vals in enumerate(raw, start=1):
            row: dict[str, float | int] = {"day": day_idx}
            for name, value in zip(FEATURE_NAMES, vals.tolist()):
                row[name] = round(float(value), 5)
            rows.append(row)
        return rows

    def predict_features(self, features: np.ndarray) -> dict[str, Any]:
        raw = self._validate_features(features)
        z_unclipped = (raw - self.feat_mean) / self.feat_std
        clipped_mask = (z_unclipped < self.clip_lo) | (z_unclipped > self.clip_hi)
        z = np.clip(z_unclipped, self.clip_lo, self.clip_hi).astype(np.float32)
        x = torch.from_numpy(z).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            p, _trajectory_internal, r = self.model(x)

        rul_raw = float(p.squeeze().cpu().item() * self.y_std + self.y_mean)
        rul_days = float(np.clip(rul_raw, RUL_MIN_DAYS, RUL_MAX_DAYS))
        rate_pct = float((r.squeeze().cpu().item() * self.r_std + self.r_mean) * 100.0)

        current_health = float(np.nanmedian(raw[-7:, 3]))
        day_axis = np.arange(0, WINDOW_SIZE + 1, dtype=np.float32)
        rate_fraction = rate_pct / 100.0
        trend_projection = current_health * (1.0 + rate_fraction * day_axis / 365.0)
        trend_end = float(trend_projection[-1])
        trend_delta = trend_end - current_health

        if rate_pct <= -3.0:
            status = "退化偏快"
        elif rate_pct <= -0.5:
            status = "存在退化"
        elif rate_pct < 0.5:
            status = "基本稳定"
        else:
            status = "需复核"

        return {
            "rul_days": round(rul_days, 1),
            "rul_years": round(rul_days / 365.0, 2),
            "rul_raw_days": round(rul_raw, 1),
            "rate_pct_per_year": round(rate_pct, 3),
            "current_pmax_ratio": round(current_health, 4),
            "status": status,
            "trend_projection_60d": [round(float(v), 5) for v in trend_projection.tolist()],
            "trend_end_pmax_ratio": round(trend_end, 4),
            "trend_delta_60d": round(float(trend_delta), 4),
            "trend_projection_note": "基于模型年化退化率和最近7天Pmax中位数的线性趋势外推，不是Mamba trajectory_head的直接未来预测。",
            "input_pmax_history": [round(float(v), 5) for v in raw[:, 3].tolist()],
            "input_rows": self._rows(raw),
            "input_clipped": bool(np.any(clipped_mask)),
            "device": str(self.device),
        }

    def predict_sample(self, sample_index: int) -> dict[str, Any]:
        raw = self.sample_raw(sample_index)
        result = self.predict_features(raw)
        target_rul = float(self.y_rul[sample_index, 0])
        target_rate = float(self.y_rate[sample_index, 0] * 100.0)
        result.update({
            "sample_index": sample_index,
            "target_rul_days": round(target_rul, 1),
            "target_rate_pct_per_year": round(target_rate, 3),
        })
        return result

    def sample_payload(self, sample_index: int) -> dict[str, Any]:
        raw = self.sample_raw(sample_index)
        return {
            "sample_index": sample_index,
            "window_days": WINDOW_SIZE,
            "features": FEATURE_NAMES,
            "rows": self._rows(raw),
        }

    def meta(self) -> dict[str, Any]:
        metrics = self.validation.get("metrics", self.validation)
        return {
            "model_name": "PV_RUL_Mamba PEFT-1403",
            "model_type": "Mamba + LoRA + physics-informed multi-task training",
            "dataset_name": "PVDAQ 1403",
            "sample_count": self.n_samples,
            "window_days": WINDOW_SIZE,
            "logical_channel_count": 1,
            "source_channel_note": "客户快照双通道为重复副本，演示固定使用第1路。",
            "feature_names": FEATURE_NAMES,
            "feature_labels": FEATURE_LABELS,
            "feature_units": FEATURE_UNITS,
            "device": str(self.device),
            "validation_metrics": metrics,
            "limitations": [
                "RUL 当前作为辅助参考值，不应作为精确更换日期承诺。",
                "业务核心模型输出为 RUL 与年化退化率；trajectory_head 主要承担训练期物理/平滑/一致性约束，不作为未来60天直接预测展示。",
                "页面未来60天虚线仅为基于年化退化率的趋势外推，并非Mamba直接预测。",
                "V1 上传接口接收已经形成的 60×6 日级模型特征，不负责现场原始 SCADA 到模型特征的全量预处理。",
                "海南原始数据和海南历史结果未进入公开仓库。",
            ],
        }
