# Very Easy Environment Experiment

## 📋 目的

在简化环境(Very Easy)中评估3个模型的表现，证明：
1. Benchmark是可完成的（所有模型能跑完30天）
2. 模型之间有性能差异
3. 建立性能基线

## 🎯 实验配置

### 环境配置
- **品类数量**: 3个（Easy是5个）
- **总SKU数**: 18个
- **初始资金**: 25,000（Easy是10,000）
- **日租金**: 80（Easy是250）
- **动态元素**: 无
- **需求波动**: 低

### 模型选择
- DeepSeek-V3.2 (开源SOTA)
- GLM-4.6 (闭源中等)
- Kimi-K2 Thinking (闭源推理)

### 运行规模
- 每个模型运行3次
- 使用不同种子: [0, 1, 2]
- 总运行数: 9次

## 🚀 运行步骤

### 第1步：配置API密钥

创建 `.env` 文件或设置环境变量：

```bash
# DeepSeek
export DEEPSEEK_API_KEY="your_deepseek_key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"

# GLM (智谱)
export GLM_API_KEY="your_glm_key"
export GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"

# Kimi (月之暗面)
export KIMI_API_KEY="your_kimi_key"
export KIMI_BASE_URL="https://api.moonshot.cn/v1"
```

### 第2步：运行实验

```bash
# 运行所有模型的实验
python run_very_easy_experiment.py

# 或者运行特定模型
python run_very_easy_experiment.py --models deepseek-v3.2 glm-4.6

# 或者使用特定种子
python run_very_easy_experiment.py --seeds 0 1 2 3 4
```

### 第3步：分析结果

```bash
python analyze_very_easy_results.py
```

## 📊 预期输出

### 成功标准

所有模型应该达到：
- ✅ **生存率**: 100% (30/30天)
- ✅ **完成率**: 100%
- ✅ **性能差异**: 20-40%差异范围

### 预期性能

| 模型 | 预期日均利润 | 预期生存率 |
|------|------------|-----------|
| DeepSeek-V3.2 | 270-300 | 100% |
| GLM-4.6 | 240-270 | 100% |
| Kimi-K2 | 210-240 | 100% |

### 输出文件

运行完成后会生成：

```
experiments/very_easy/results/
├── very_easy_summary.json          # 完整结果（JSON格式）
├── very_easy_results.csv            # 结果表格（CSV格式）
├── comparison_table.csv             # 对比表格（用于论文）
├── latex_table.tex                  # LaTeX表格（直接用于论文）
├── comparison_figure.pdf            # 对比图
├── deepseek-v3.2_seed0_trajectory.json  # 详细轨迹（种子0）
├── deepseek-v3.2_seed1_trajectory.json  # 详细轨迹（种子1）
├── deepseek-v3.2_seed2_trajectory.json  # 详细轨迹（种子2）
├── glm-4.6_seed0_trajectory.json
├── glm-4.6_seed1_trajectory.json
├── glm-4.6_seed2_trajectory.json
├── kimi-k2-thinking_seed0_trajectory.json
├── kimi-k2-thinking_seed1_trajectory.json
└── kimi-k2-thinking_seed2_trajectory.json
```

## 🔍 故障排查

### 问题1：API密钥错误

```
Error: Authentication failed
```

**解决**：
- 检查API密钥是否正确
- 检查API基础URL是否正确
- 确认账户有足够余额

### 问题2：模块导入错误

```
ModuleNotFoundError: No module named 'xxx'
```

**解决**：
```bash
pip install openai pandas scipy matplotlib
```

### 问题3：环境配置错误

```
Error: Unknown config type 'very_easy'
```

**解决**：
- 确认 `util/very_easy_config.py` 已创建
- 确认在 `util/default_config.py` 中添加了 very_easy 配置

### 问题4：模型运行失败

```
Error: Model failed to complete episode
```

**解决**：
- 检查日志文件：`logs/` 或 `model_run_time/`
- 查看具体错误信息
- 尝试降低环境复杂度（使用方案B配置）

## 📈 下一步

实验完成后：

1. **验证结果**
   - 检查所有模型是否100%生存
   - 检查模型之间是否有显著差异

2. **更新论文**
   - 添加环境校准章节
   - 添加对比表格
   - 更新abstract和conclusion

3. **进行消融实验**
   - 使用相同的模型和种子
   - 对比不同框架配置

## 💰 成本估算

| 模型 | 单次成本 | 3次运行 | 备注 |
|------|---------|---------|------|
| DeepSeek-V3.2 | ~$0.5-1 | $1.5-3 | 最便宜 |
| GLM-4.6 | ~$1-2 | $3-6 | 中等 |
| Kimi-K2 | ~$2-3 | $6-9 | 最贵 |
| **总计** | - | **$10-18** | - |

## ⏱️ 时间估算

- 单次运行：30-60分钟（取决于模型速度和网络）
- 9次运行总计：约5-10小时
- 建议：使用并行运行（如果有多个API密钥）

## 📞 支持

如有问题，请检查：
1. API密钥配置
2. 网络连接
3. 日志文件
4. 错误信息
