# Runtime assets

这里的 `.npy/.pth` 运行资产不直接提交到公开 GitHub。

请从客户提供的 `光伏组件RUL预测.zip` 导入：

```bash
python scripts/import_assets.py "/path/to/光伏组件RUL预测.zip"
```

导入脚本只复制 V1 需要的 **DKASC scaler + PVDAQ 1403 模型/样本/标签**，明确不复制：

- `data/hainan_*`
- `checkpoints/hainan_results/*`

这是为了遵守原研究快照中对海南材料“涉密，勿 git add”的说明。
