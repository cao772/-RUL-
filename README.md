# 光伏组件健康状态与 RUL 预测平台（V1 Demo）

这是基于客户提供研究快照整理出的**可本地复跑演示系统**。V1 目标不是重新训练模型，而是先打通：

> 真实 60 天模型输入 → 本地 PyTorch / Mamba 推理 → RUL / 年化退化率 / 未来 60 天 Pmax 轨迹 → 前端可视化

## 当前已经能做什么

- 内置 PVDAQ 1403 的 740 个真实 60 天窗口，可在前端选择并实时推理。
- 使用 `PV_RUL_Mamba PEFT-1403` 完整权重，不是写死结果。
- 输出：RUL 参考值、年化退化率、当前 Pmax、未来 60 天 Pmax 轨迹。
- 展示 6 个输入特征的 60 天曲线。
- 支持上传标准模型特征 CSV（必须 60 行 × 6 列）并调用同一个模型推理。
- 展示客户包内已有验证指标。

## 当前边界

1. **V1 上传的是模型特征，不是现场原始 SCADA 数据。** 当前客户尚未确认现场功率层级、额定功率、停机/限电状态、STC 修正等生产口径，因此暂不把原始数据适配逻辑写死。
2. **RUL 是辅助指标。** 客户包内验证报告给出的 RUL `R²=0.5597`、`MAE=2022.4 天`，演示中不要把 RUL 解释成精确更换日期。
3. **海南原始数据与海南历史结果没有上传到本公开仓库。** 原研究 README 明确标注这些材料为涉密，因此仓库只使用非海南的 PVDAQ 1403 演示资产。
4. V1 不重新训练模型；训练原始数据并未完整包含在客户快照中。

## 模型输入

每次输入固定为 `60 × 6`：

| 字段 | 含义 | 单位 |
|---|---|---|
| `power_ratio` | 功率比 | p.u. |
| `irradiance` | 辐照度 | W/m² |
| `ambient_temp` | 环境温度 | ℃ |
| `pmax_ratio` | Pmax 健康指数 | p.u. |
| `module_temp` | 组件温度 | ℃ |
| `rh` | 相对湿度 | % |

标准 CSV 示例见：`app/static/sample_feature_60d.csv`。

## 模型输出

- `rul_days`：RUL 参考天数（对外约束在 30~10950 天）
- `rate_pct_per_year`：年化退化率 `%/年`
- `trajectory`：未来 60 天 Pmax 健康轨迹
- `current_pmax_ratio`：输入最近 7 天 Pmax 中位数
- `forecast_end_pmax_ratio`：未来 60 天轨迹末值
- `status`：基于退化率的演示性状态标签

## 本地启动

### 0. 导入客户包中的运行资产

仓库是公开仓库，因此模型权重和训练窗口不直接提交。clone 后先执行：

```bash
python scripts/import_assets.py "/path/to/光伏组件RUL预测.zip"
```

该脚本只导入 DKASC scaler 与 PVDAQ 1403 的模型/窗口/标签，**不会导入海南原始数据或海南历史结果**。

随后执行 Smoke test：

```bash
python scripts/smoke_test.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

也可以：

```bash
bash start.sh
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

或双击 / 执行：

```bat
start.bat
```

> 首次安装 PyTorch 体积较大。如果机器已经有可用 PyTorch 环境，可直接复用该环境安装其余依赖。

## Smoke test

```bash
python scripts/smoke_test.py
```

正常结束会看到：

```text
SMOKE_OK
```

## API

启动后可访问 FastAPI 自动文档：

```text
http://127.0.0.1:8000/docs
```

主要接口：

- `GET /api/health`
- `GET /api/meta`
- `GET /api/sample/{sample_index}?channel=0`
- `POST /api/predict/sample`
- `POST /api/predict/features`
- `POST /api/predict/csv`

## 目录

```text
.
├── app/
│   ├── main.py                 FastAPI 接口
│   ├── inference.py            真实模型加载与推理
│   ├── model/model_common.py   Mamba / GRN / LoRA 模型定义
│   └── static/                 无外部前端依赖的可视化页面
├── assets/checkpoints/          本地导入的运行资产（Git 忽略二进制）
├── scripts/import_assets.py     从客户 ZIP 安全导入非海南资产
├── scripts/smoke_test.py
├── requirements.txt
└── run.py
```

## 下一阶段

拿到客户现场原始样例后，补充一层：

```text
现场原始数据
  ↓
数据质检 / 停机限电处理
  ↓
STC 修正 / Pmax 基线构造
  ↓
60×6 模型特征
  ↓
V1 现有推理接口
```

这样不用推翻 V1 的模型服务与前端。
