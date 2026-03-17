#!/bin/bash
# 零售环境模拟器 - 并行执行脚本

# 设置日志目录，避免多个进程写入同一个日志文件
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="logs/parallel_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "开始并行执行所有模拟任务"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

# 1. 运行工具自检（通常很快，可以单独运行或注释掉）
# echo "=== 启动工具自检 ==="
# python3 retail_environment.py --mode tools > "${LOG_DIR}/tools.log" 2>&1 &
# TOOLS_PID=$!

# 2. 运行简单逻辑环境（1460天）
echo "=== 启动简单逻辑环境（1460天） ==="
python3 retail_environment.py --mode logic --days 1460 --sample-size 2 --db-path middle/simulate_data/15/simple/ > "${LOG_DIR}/logic.log" 2>&1 &
LOGIC_PID=$!
echo "  PID: $LOGIC_PID, 日志: ${LOG_DIR}/logic.log"
sleep 5

# 3. 运行评价感知环境（1460天）
echo "=== 启动评价感知环境（1460天） ==="
python3 retail_environment.py --mode review --days 1460 --sample-size 2 --db-path middle/simulate_data/15/records_review/ > "${LOG_DIR}/review.log" 2>&1 &
REVIEW_PID=$!
echo "  PID: $REVIEW_PID, 日志: ${LOG_DIR}/review.log"
sleep 5

# 4. 运行新闻感知环境（1460天）
echo "=== 启动新闻感知环境（1460天） ==="
python3 retail_environment.py --mode news --days 1460 --sample-size 2 --db-path middle/simulate_data/15/records_review_news/ > "${LOG_DIR}/news.log" 2>&1 &
NEWS_PID=$!
echo "  PID: $NEWS_PID, 日志: ${LOG_DIR}/news.log"

echo ""
echo "=========================================="
echo "所有任务已启动，正在并行执行..."
echo "=========================================="
echo ""
echo "监控进程状态："
echo "  ps aux | grep 'retail_environment.py' | grep -v grep"
echo ""
echo "查看实时日志："
echo "  tail -f ${LOG_DIR}/logic.log"
echo "  tail -f ${LOG_DIR}/review.log"
echo "  tail -f ${LOG_DIR}/news.log"
echo ""
echo "等待所有任务完成..."

# 等待所有后台任务完成
wait $LOGIC_PID
LOGIC_EXIT=$?
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 简单逻辑环境完成 (退出码: $LOGIC_EXIT)"

wait $REVIEW_PID
REVIEW_EXIT=$?
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 评价感知环境完成 (退出码: $REVIEW_EXIT)"

wait $NEWS_PID
NEWS_EXIT=$?
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 新闻感知环境完成 (退出码: $NEWS_EXIT)"

echo ""
echo "=========================================="
echo "所有任务执行完成！"
echo "=========================================="
echo "退出码汇总:"
echo "  简单逻辑环境: $LOGIC_EXIT"
echo "  评价感知环境: $REVIEW_EXIT"
echo "  新闻感知环境: $NEWS_EXIT"
echo ""
echo "日志文件位置: $LOG_DIR"

