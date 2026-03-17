#!/bin/bash
# 并行启动多个命令行，每个进程间隔指定时间启动
#
# 用法:
#   ./run_parallel_commands.sh --config config.txt --interval 5
#
# 配置文件格式（每行一个命令）:
#   python run_env.py --model model1 --config_type dynamic_hard
#   python run_env.py --model model2 --config_type dynamic_hard
#   python run_env.py --model model3 --config_type dynamic_hard

# 注意：不使用 set -e，因为后台进程的 wait 可能返回非零退出码

# 默认值
INTERVAL=5
CONFIG_FILE=""
WAIT=false
CWD=""
LOG_DIR=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印帮助信息
print_help() {
    cat << EOF
用法: $0 [选项]

选项:
    --config FILE               从配置文件加载命令（每行一个命令）
    --interval SECONDS          启动间隔（秒），默认 5
    --cwd DIR                   工作目录（所有命令在此目录下执行）
    --wait                      等待所有进程完成后再退出
    --log-dir DIR               日志目录（可选，用于保存进程输出）
    -h, --help                  显示此帮助信息

示例:
    # 从配置文件加载
    $0 --config commands.txt --interval 5
    
    # 等待所有进程完成
    $0 --config commands.txt --wait
    
    # 指定工作目录和日志目录
    $0 --config commands.txt --cwd /path/to/workdir --log-dir /path/to/logs
EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --interval)
                INTERVAL="$2"
                shift 2
                ;;
            --cwd)
                CWD="$2"
                shift 2
                ;;
            --wait)
                WAIT=true
                shift
                ;;
            --log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                echo -e "${RED}错误: 未知参数 $1${NC}" >&2
                print_help
                exit 1
                ;;
        esac
    done
}

# 从配置文件加载命令
load_commands_from_file() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}" >&2
        exit 1
    fi
    
    COMMANDS=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        # 跳过空行和注释行
        if [[ -n "$line" ]] && [[ ! "$line" =~ ^[[:space:]]*# ]]; then
            COMMANDS+=("$line")
        fi
    done < "$CONFIG_FILE"
}

# 清理函数（在退出时调用）
cleanup() {
    echo -e "\n${YELLOW}[中断] 收到中断信号，正在终止所有进程...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}终止进程 PID $pid...${NC}"
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # 等待进程结束
    sleep 2
    
    # 强制终止仍在运行的进程
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}强制终止进程 PID $pid...${NC}"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    echo -e "${GREEN}所有进程已终止${NC}"
    exit 1
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    # 检查参数
    if [[ -z "$CONFIG_FILE" ]]; then
        echo -e "${RED}错误: 必须指定 --config${NC}" >&2
        print_help
        exit 1
    fi
    
    # 从配置文件加载命令
    load_commands_from_file
    
    if [[ ${#COMMANDS[@]} -eq 0 ]]; then
        echo -e "${RED}错误: 配置文件中没有找到要执行的命令${NC}" >&2
        exit 1
    fi
    
    # 验证工作目录
    if [[ -n "$CWD" ]] && [[ ! -d "$CWD" ]]; then
        echo -e "${RED}错误: 工作目录不存在: $CWD${NC}" >&2
        exit 1
    fi
    
    # 创建日志目录
    if [[ -n "$LOG_DIR" ]]; then
        mkdir -p "$LOG_DIR"
    fi
    
    # 显示信息
    echo -e "${BLUE}$(printf '=%.0s' {1..80})${NC}"
    echo -e "${BLUE}准备启动 ${#COMMANDS[@]} 个进程，间隔 ${INTERVAL} 秒${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..80})${NC}"
    echo ""
    
    # 存储所有进程ID
    PIDS=()
    
    # 启动每个命令
    for i in "${!COMMANDS[@]}"; do
        cmd="${COMMANDS[$i]}"
        num=$((i + 1))
        
        echo -e "${GREEN}[启动] 进程 $num/${#COMMANDS[@]}${NC}"
        echo -e "  命令: ${YELLOW}$cmd${NC}"
        if [[ -n "$CWD" ]]; then
            echo -e "  工作目录: $CWD"
        fi
        
        # 构建日志文件路径
        if [[ -n "$LOG_DIR" ]]; then
            log_file="$LOG_DIR/process_${num}.log"
            echo -e "  日志文件: $log_file"
        fi
        
        # 切换到工作目录（如果指定）
        if [[ -n "$CWD" ]]; then
            cd "$CWD" || exit 1
        fi
        
        # 启动进程
        if [[ -n "$LOG_DIR" ]]; then
            # 将输出重定向到日志文件
            eval "$cmd" > "$log_file" 2>&1 &
        else
            # 输出到终端
            eval "$cmd" &
        fi
        
        PID=$!
        PIDS+=("$PID")
        
        echo -e "  进程 ID: ${GREEN}$PID${NC}"
        echo ""
        
        # 如果不是最后一个，等待指定间隔
        if [[ $i -lt $((${#COMMANDS[@]} - 1)) ]]; then
            echo -e "${BLUE}等待 ${INTERVAL} 秒后启动下一个进程...${NC}"
            echo ""
            sleep "$INTERVAL"
        fi
    done
    
    # 显示所有进程ID
    echo -e "${BLUE}$(printf '=%.0s' {1..80})${NC}"
    echo -e "${GREEN}已启动 ${#PIDS[@]} 个进程${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..80})${NC}"
    echo ""
    echo -e "${GREEN}进程列表:${NC}"
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        num=$((i + 1))
        cmd="${COMMANDS[$i]}"
        echo -e "  ${num}. PID ${GREEN}$pid${NC}: ${YELLOW}${cmd:0:60}...${NC}"
    done
    echo ""
    
    if [[ "$WAIT" == true ]]; then
        echo -e "${BLUE}等待所有进程完成...${NC}"
        echo ""
        
        # 等待所有进程完成
        for i in "${!PIDS[@]}"; do
            pid="${PIDS[$i]}"
            num=$((i + 1))
            cmd="${COMMANDS[$i]}"
            
            echo -e "${BLUE}等待进程 $num (PID $pid) 完成...${NC}"
            if wait "$pid" 2>/dev/null; then
                exit_code=$?
                if [[ $exit_code -eq 0 ]]; then
                    echo -e "${GREEN}进程 $num 已完成，退出码: 0${NC}"
                else
                    echo -e "${YELLOW}进程 $num 已完成，退出码: $exit_code${NC}"
                fi
            else
                exit_code=$?
                echo -e "${YELLOW}进程 $num 已完成，退出码: $exit_code${NC}"
            fi
            echo ""
        done
        
        echo -e "${GREEN}所有进程已完成${NC}"
    else
        echo -e "${BLUE}所有进程已在后台启动，主进程退出${NC}"
        echo -e "${YELLOW}提示: 使用 --wait 参数可以等待所有进程完成${NC}"
        echo ""
        echo -e "${BLUE}要查看进程状态，使用:${NC}"
        echo -e "  ps -p ${PIDS[*]}"
        echo ""
        echo -e "${BLUE}要终止所有进程，使用:${NC}"
        echo -e "  kill ${PIDS[*]}"
    fi
}

# 运行主函数
main "$@"
