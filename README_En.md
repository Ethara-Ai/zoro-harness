# RetailBench

RetailBench is a retail operation simulation and agent evaluation project.  
Its main goal is to simulate daily store operations (ordering, pricing, inventory decisions) under historical data, supplier dynamics, reviews, and news impact, then compare different agent strategies.

## 1. Project Structure

```text
RetailBench/
├── retail_environment.py            # Core simulator (tools, day-step logic, CLI modes)
├── run_env.py                       # Two-phase agent (Strategy + Execution)
├── run_plan_and_act.py              # Plan-and-Act agent
├── run_step_reflection.py           # Step-level reflection agent
├── run_reflection.py                # Execution + daily reflection memory
├── run_exec_strategy_env.py         # Fixed-strategy replay (execution only)
├── stream_chat.py                   # Streaming LLM call wrapper with retries
├── inventory.py                     # Inventory, sales, returns, expiration handling
├── sku.py                           # SKU demand/attraction modeling
├── module/                          # Business managers (orders/suppliers/reviews/news/records/strategy)
├── model/                           # Rating/return-rate/demand model implementations
├── util/                            # Default configs, logging, SQL formatting, tool-call parser
├── data/                            # Simulation input datasets (dynamic / still)
├── data_process/                    # Data preprocessing and generation scripts
├── paper_data/                      # Experiment result dataset (scenario/model/run layout)
├── env_data/                        # easy/middle/hard environment comparison data
├── analysis/                        # Analysis and plotting scripts (see Section 5)
├── script/                          # Batch run command configs
├── logs/                            # Run logs
└── model_run_time/                  # Runtime databases and result artifacts
```

## 2. Main Entry Points

### 2.1 `retail_environment.py` (non-LLM baseline simulation)

Built-in environment modes:

- `--mode tools`: tool self-check
- `--mode logic`: simple logic environment
- `--mode review`: review-aware environment
- `--mode news`: news-aware environment
- `--mode quality`: quality-priority environment

Example:

```bash
python3 retail_environment.py --mode logic --days 30 --sample-size 2 --config-type still_middle
```

### 2.2 `run_env.py` (recommended primary entry)

Two-phase agent loop:

1. Strategy Phase: analyze and set strategy
2. Execution Phase: execute tool actions and call `end_today`

Example:

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

### 2.3 Other agent runners

- `run_plan_and_act.py`: generate a daily plan, then execute
- `run_step_reflection.py`: reflect after each step
- `run_reflection.py`: accumulate reflection memory by day
- `run_exec_strategy_env.py`: execute with a fixed strategy from `day_*_final_strategy.json`

Fixed-strategy replay example:

```bash
python3 run_exec_strategy_env.py \
  --strategy_file logs/run_env_xxx/day_1_final_strategy.json \
  --model qwen3-235b-a22b-thinking-2507 \
  --config_type still_middle \
  --db_path model_run_time/fixed_strategy_run \
  --api_key "<YOUR_API_KEY>" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## 3. Setup

### 3.1 Python and virtual environment

Python 3.10+ is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

There is no unified `requirements.txt` currently. Start with:

```bash
pip install openai numpy pandas matplotlib scipy tqdm requests beautifulsoup4 tabulate
```

Notes:

- `run_*.py` requires at least `openai` and `numpy`
- analysis plotting requires `matplotlib` and `scipy`
- some statistics scripts optionally use `tabulate`

### 3.3 Config types

`--config_type` supports:

- `dynamic_hard`
- `dynamic_middle`
- `still_hard`
- `still_middle`

Definitions are in `util/default_config.py`. Main differences include:

- data source (`data/dynamic/...` vs `data/still/...`)
- store scale (funds/rent/capacity/category count)
- whether review/news effects are enabled

## 4. Outputs and Recovery

Each run creates a timestamped directory under `logs/`, typically containing:

- `config.json`: effective runtime config
- `args.json`: command arguments
- `token_statistics.json`: token usage summary
- `checkpoints/`: turn/day checkpoints
- `day_*_final_strategy.json`: daily final strategy (`run_env.py`)
- `tool_calls.jsonl`: tool call records

Recovery flags:

- `--recover_turn N`: resume from a specific turn
- `--recover_day N`: resume from a day-level checkpoint (for supported runners)

## 5. `analysis/` Files: What Each File Does

### 5.1 `analysis/analyze_experiment_data/analyze_paper_data_final.py`

- Parses multi-scenario, multi-model runs under `paper_data/`
- Extracts daily sales/profit/net worth/expiry/return metrics from `tool_calls.jsonl`
- Produces summary tables and JSON outputs
- Generates paper-style plots (net worth curves, category metrics bars, selected-model comparison)

### 5.2 `analysis/analyze_strategy_similarity/analyze_strategy_similarity.py`

- Reads `day_*_final_strategy.json`
- Computes strategy similarity:
  - `macro_strategy`: LLM-based similarity score
  - `execute_strategy`: set intersection/union score
  - `both`: combined score
- Supports multiprocessing over `paper_data/`
- Outputs `strategy_analysis/strategy_similarity_analysis.json`

### 5.3 `analysis/plot_strategy_similarity/plot_strategy_similarity.py`

- Reads strategy similarity JSON
- Aggregates curves by scenario/model
- Supports smoothing and weighted score (`w_macro`, `w_exec`)
- Exports PNG/PDF plots and optional statistics

### 5.4 `analysis/analyze_tool_uses/check_tool_calls.py`

- Scans `paper_data/**/tool_calls.jsonl`
- Checks problematic calls:
  - `modify_sku_price` price <= 0 or > 50
  - `place_order` single-SKU quantity > 2000
- Writes issue details to `tool_calls_issues.json`

### 5.5 `analysis/analyze_tool_uses/aggregate_focus_sku_tools_by_model.py`

- Aggregates from `focus_sku_tools_daily.json`
- Computes model-by-scenario focus-SKU tool usage metrics:
  - `avg_calls_per_sku_day`
  - `usage_frequency`
  - `avg_calls_per_using_sku`
- Outputs aggregated JSON and terminal tables

### 5.6 `analysis/analysis_strategy_focus_skus_data/analyze_focus_sku_tools_daily.py`

- Parses strategy-phase tool calls by run/day
- Extracts `focus_skus` and related tool call counts
- Generates `focus_sku_tools_daily.json` (input for downstream aggregation/plotting)

### 5.7 `analysis/analysis_strategy_focus_skus_data/plot_focus_sku_tools_vs_performance.py`

- Reads `focus_sku_tools_daily.json` and `paper_data/*/*/*/tool_calls.jsonl`
- Analyzes relationship between focus-SKU tool intensity and performance (sales/income)
- Exports per-tool scatter plots, trend lines, and optional correlations

### 5.8 `analysis/analyze_env_heutrial_data/analyze_env_data.py`

- Aggregates `env_data/easy|middle|hard` from `tool_calls.jsonl`
- Outputs scenario-level metrics (sales, profit, expiry ratio, return ratio, etc.)
- Writes `env_data_analysis/statistics.json`

### 5.9 `analysis/plot_env_data/plot_env_data_metrics.py`

- Plots time-series metrics from `env_data/`:
  - net worth
  - money balance
  - cumulative units sold
  - cumulative expired items
- Exports single plots and a 2x2 summary figure in PNG/PDF

### 5.10 `analysis/analysis_strategy_focus_skus_data/focus_sku_tools_daily.json`

- Pre-generated intermediate analysis output
- Produced by `analyze_focus_sku_tools_daily.py`

### 5.11 `analysis/analysis_strategy_focus_skus_data/tool_calls_issues.json`

- Pre-generated issue report
- Produced by `check_tool_calls.py`

### 5.12 `analysis/analyze_experiment_data/__pycache__/analyze_paper_data_final.cpython-310.pyc`

- Auto-generated Python bytecode cache (not hand-written business code)

## 6. Batch Run Utilities

- `run_parallel_commands.sh`: launch commands from a config file with interval/log/wait options
- `script/run_plan_and_act.sh`: Plan-and-Act batch command examples
- `script/run_step_reflection.sh`: Step-Reflection batch command examples
- `script/run_gpt5_2.sh`: model-specific batch command examples
- `script/run_sota.sh`: currently empty

Example:

```bash
bash run_parallel_commands.sh --config script/run_plan_and_act.sh --interval 5 --wait
```

## 7. Recommended Minimal Repro Flow

1. Install dependencies (Section 3)
2. Validate environment quickly:

```bash
python3 retail_environment.py --mode tools
python3 retail_environment.py --mode logic --days 7 --sample-size 2 --config-type still_middle
```

3. Run an agent:

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

4. Optional analysis:

```bash
python3 analysis/analyze_experiment_data/analyze_paper_data_final.py
```

