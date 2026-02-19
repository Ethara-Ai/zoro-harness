from __future__ import annotations

import logging
import sys
from typing import Optional

_LOGGER_NAME = "RetailBench"

logger = logging.getLogger(_LOGGER_NAME)

def get_logger() -> logging.Logger:
    """
    返回全局 logger；可传 debug=True/False 调整等级。
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

def set_logger_level(debug: Optional[bool] = None) -> None:
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    # 同时更新所有 handler 的级别，确保日志级别设置生效
    for handler in logger.handlers:
        handler.setLevel(level)
