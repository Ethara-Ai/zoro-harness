# 最常见：每 5 秒启动一条
bash run_parallel_commands.sh --config script/run_plan_and_act.sh --interval 5 --wait

bash run_parallel_commands.sh --config script/run_step_reflection.sh --interval 5 --wait

bash run_parallel_commands.sh --config script/run_sota.sh --interval 5 --wait

bash run_parallel_commands.sh --config script/run_gpt5_2.sh --interval 5 --wait


sudo pmset disablesleep 1

python run_plan_and_act.py --model kimi-k2-thinking --db_path model_run_time/kimi-k2_5_run1 --config_type still_middle --max_input_tokens 40000 --max_days 180 --max_turns 5 --max_execution_turns 10 --api_key sk-b0ba29e85cac4cf9bfcb24d3a482cd17 --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 > kimi-k2_5_run1.log
python run_plan_and_act.py --model kimi-k2-thinking --db_path model_run_time/kimi-k2_5_run2 --config_type still_middle --max_input_tokens 40000 --max_days 180 --max_turns 5 --max_execution_turns 10 --api_key sk-b0ba29e85cac4cf9bfcb24d3a482cd17 --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 > kimi-k2_5_run2.log
python run_plan_and_act.py --model kimi-k2-thinking --db_path model_run_time/kimi-k2_5_run3 --config_type still_middle --max_input_tokens 40000 --max_days 180 --max_turns 5 --max_execution_turns 10 --api_key sk-b0ba29e85cac4cf9bfcb24d3a482cd17 --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 > kimi-k2_5_run3.log