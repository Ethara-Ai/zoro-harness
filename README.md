# RetailBench

RetailBench 是一个零售经营仿真与智能体评测项目。  
核心目标是：在给定历史数据、供应商、评价和新闻影响下，模拟门店每天的补货、定价和经营结果，并对不同 Agent 策略进行对比分析。

## 1. 项目结构

```text
RetailBench/
├── retail_environment.py            # 核心仿真环境（工具定义、日推进、CLI模式）
├── run_env.py                       # 双阶段 Agent（Strategy + Execution）
├── run_plan_and_act.py              # Plan-and-Act Agent
├── run_step_reflection.py           # Step-level Reflection Agent
├── run_reflection.py                # Execution + 日级反思记忆
├── run_exec_strategy_env.py         # 固定策略回放（只执行）
├── stream_chat.py                   # LLM流式调用与重试封装
├── inventory.py                     # 库存与销售/退货/过期处理
├── sku.py                           # SKU需求与吸引力模型
├── module/                          # 业务管理模块（订单/供应商/评价/新闻/记录/策略）
├── model/                           # 评分/退货/需求等模型实现
├── util/                            # 默认配置、日志、SQL格式化、工具调用解析
├── data/                            # 仿真输入数据（dynamic / still）
├── data_process/                    # 数据预处理与生成脚本
├── paper_data/                      # 论文实验结果数据集（按场景/模型/run组织）
├── env_data/                        # easy/middle/hard 环境对照数据
├── analysis/                        # 实验分析与画图脚本（见第5节）
├── script/                          # 批量运行命令配置
├── logs/                            # 运行日志输出目录
└── model_run_time/                  # 运行时数据库与结果目录
```

## 2. 核心运行入口

### 2.1 `retail_environment.py`（非LLM基线模拟）

用于直接运行环境内置逻辑：

- `--mode tools`：工具自检
- `--mode logic`：简单逻辑环境
- `--mode review`：评价感知环境
- `--mode news`：新闻感知环境
- `--mode quality`：质量优先环境

示例：

```bash
python3 retail_environment.py --mode logic --days 30 --sample-size 2 --config-type still_middle
```

### 2.2 `run_env.py`（推荐主入口）

双阶段智能体：

1. Strategy Phase：分析并设置策略
2. Execution Phase：执行工具动作并 `end_today`

示例：

```bash
python3 run_env.py \
  --model qwen3-235b-a22b-thinking-2507 \
  --config_type still_middle \
  --db_path model_run_time/demo_run_env \
  --max_days 30 \
  --max_strategy_turns 10 \
  --max_execution_turns 20 \
  --api_key "<YOUR_API_KEY>" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 2.3 其他 Agent 入口

- `run_plan_and_act.py`：先给每日计划，再执行
- `run_step_reflection.py`：每步执行后即时反思
- `run_reflection.py`：按天积累反思记忆
- `run_exec_strategy_env.py`：读取 `day_*_final_strategy.json` 固定策略执行

固定策略执行示例：

```bash
python3 run_exec_strategy_env.py \
  --strategy_file logs/run_env_xxx/day_1_final_strategy.json \
  --model qwen3-235b-a22b-thinking-2507 \
  --config_type still_middle \
  --db_path model_run_time/fixed_strategy_run \
  --api_key "<YOUR_API_KEY>" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## 3. 启动前准备

### 3.1 Python 与虚拟环境

推荐 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 安装依赖

仓库当前没有统一 `requirements.txt`，可先安装核心依赖：

```bash
pip install openai numpy pandas matplotlib scipy tqdm requests beautifulsoup4 tabulate
```

说明：

- 运行 `run_*.py` 至少需要 `openai`、`numpy`
- 分析绘图需要 `matplotlib`、`scipy`
- 部分统计表格脚本会用到 `tabulate`

### 3.3 配置类型

`--config_type` 支持：

- `dynamic_hard`
- `dynamic_middle`
- `still_hard`
- `still_middle`

对应配置定义在 `util/default_config.py`，主要区别是：

- 数据源（`data/dynamic/...` vs `data/still/...`）
- 门店规模（资金/租金/容量/品类数量）
- 是否启用 review/news

## 4. 运行产物与恢复

每次运行会在 `logs/` 下创建时间戳目录，常见内容：

- `config.json`：本次实际配置
- `args.json`：命令参数
- `token_statistics.json`：总 token 统计
- `checkpoints/`：按 turn/day checkpoint
- `day_*_final_strategy.json`：每日最终策略（`run_env.py`）
- `tool_calls.jsonl`：工具调用记录

恢复运行：

- `--recover_turn N`：从某个 turn 恢复
- `--recover_day N`：从某天 checkpoint 恢复（支持脚本内定义的入口）

## 5. `analysis/` 目录文件用途说明

当前 `analysis/` 下实际文件如下：

### 5.1 `analysis/analyze_experiment_data/analyze_paper_data_final.py`

- 读取 `paper_data/` 多场景多模型 run 结果
- 提取 `tool_calls.jsonl` 中的日销售、利润、净值、过期率、退货率
- 统计并输出表格/JSON
- 生成论文风格图（净值轨迹、分类指标柱状图、选定模型对比）

### 5.2 `analysis/analyze_strategy_similarity/analyze_strategy_similarity.py`

- 读取 `day_*_final_strategy.json`
- 计算策略相似度：
  - `macro_strategy`：通过 LLM 评分相似度
  - `execute_strategy`：集合交并比
  - `both`：综合相似度
- 支持多进程批量分析 `paper_data/`
- 输出 `strategy_analysis/strategy_similarity_analysis.json`

### 5.3 `analysis/plot_strategy_similarity/plot_strategy_similarity.py`

- 读取策略相似度分析结果 JSON
- 按场景/模型聚合为曲线
- 支持平滑与 weighted score（`w_macro`, `w_exec`）
- 输出 PNG/PDF 图，支持附加统计指标计算

### 5.4 `analysis/analyze_tool_uses/check_tool_calls.py`

- 扫描 `paper_data/**/tool_calls.jsonl`
- 检查异常工具调用：
  - `modify_sku_price` 价格 <=0 或 >50
  - `place_order` 单 SKU 下单量 >2000
- 输出详细问题到 `tool_calls_issues.json`

### 5.5 `analysis/analyze_tool_uses/aggregate_focus_sku_tools_by_model.py`

- 基于 `focus_sku_tools_daily.json` 聚合
- 计算每个模型在各场景中，focus SKU 的工具调用特征：
  - `avg_calls_per_sku_day`
  - `usage_frequency`
  - `avg_calls_per_using_sku`
- 输出聚合 JSON + 终端统计表

### 5.6 `analysis/analysis_strategy_focus_skus_data/analyze_focus_sku_tools_daily.py`

- 按 run/day 读取 strategy 阶段调用
- 提取 `focus_skus` 以及关联的工具调用次数
- 生成 `focus_sku_tools_daily.json`（后续聚合和画图的输入）

### 5.7 `analysis/analysis_strategy_focus_skus_data/plot_focus_sku_tools_vs_performance.py`

- 读取 `focus_sku_tools_daily.json` + `paper_data/*/*/*/tool_calls.jsonl`
- 分析“focus SKU 工具调用强度”与“销售/收益”关系
- 输出每类工具对应的散点图 + 趋势线 + 相关系数（可选）

### 5.8 `analysis/analyze_env_heutrial_data/analyze_env_data.py`

- 对 `env_data/easy|middle|hard` 的 `tool_calls.jsonl` 做统计
- 输出每个场景的销售、利润、过期率、退货率等汇总
- 生成 `env_data_analysis/statistics.json`

### 5.9 `analysis/plot_env_data/plot_env_data_metrics.py`

- 对 `env_data/` 绘制时间序列图：
  - net worth
  - money balance
  - cumulative units sold
  - cumulative expired items
- 输出单图 + 2x2 总图，PNG/PDF

### 5.10 `analysis/analysis_strategy_focus_skus_data/focus_sku_tools_daily.json`

- 已生成的中间分析结果文件
- 来源：`analyze_focus_sku_tools_daily.py` 输出

### 5.11 `analysis/analysis_strategy_focus_skus_data/tool_calls_issues.json`

- 已生成的问题清单文件
- 来源：`check_tool_calls.py` 输出

### 5.12 `analysis/analyze_experiment_data/__pycache__/analyze_paper_data_final.cpython-310.pyc`

- Python 自动生成的字节码缓存，不是手写业务代码

## 6. 批量运行脚本

- `run_parallel_commands.sh`：按配置文件逐条并行启动命令（支持间隔、日志、等待）
- `script/run_plan_and_act.sh`：Plan-and-Act 批量命令示例
- `script/run_step_reflection.sh`：Step-Reflection 批量命令示例
- `script/run_gpt5_2.sh`：特定模型批量命令示例
- `script/run_sota.sh`：当前为空

示例：

```bash
bash run_parallel_commands.sh --config script/run_plan_and_act.sh --interval 5 --wait
```

## 7. 推荐最小可复现流程

1. 安装依赖（第3节）
2. 先跑环境模式验证数据可用：

```bash
python3 retail_environment.py --mode tools
python3 retail_environment.py --mode logic --days 7 --sample-size 2 --config-type still_middle
```

3. 再跑 Agent：

```bash
python3 run_plan_and_act.py \
  --model qwen3-235b-a22b-thinking-2507 \
  --config_type still_middle \
  --db_path model_run_time/demo_plan_act \
  --max_days 7 \
  --max_turns 5 \
  --max_execution_turns 10 \
  --api_key "<YOUR_API_KEY>" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

4. 分析结果（可选）：

```bash
python3 analysis/analyze_experiment_data/analyze_paper_data_final.py
```
