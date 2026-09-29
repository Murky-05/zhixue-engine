# -*- coding: utf-8 -*-
"""
log_config —— 全局日志配置（文件日志系统）
==========================================
职责：为全项目提供两个独立的文件日志器，与终端 print / 数据库 event_log 互补：

  business_logger -> logs/business.log   INFO 级：业务操作流水
                     （登录/上传/删除/AI 调用成功等，谁在什么时候做了什么）
  error_logger    -> logs/error.log      ERROR 级：完整报错堆栈
                     （任何被捕获的异常都应写入这里，前端只显示友好文案）

设计要点：
  - RotatingFileHandler：单文件超过 5MB 自动轮转（business.log.1 ...），防止无限膨胀；
  - propagate=False：不向 root logger 传播，避免与 basicConfig 的终端输出重复；
  - 幂等初始化：Streamlit 每次脚本运行都会重新 import，handler 只挂一次；
  - logs/ 目录创建失败（只读环境等）时降级为 NullHandler，绝不影响主流程。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

_FMT = "%(asctime)s [%(levelname)s] %(message)s"   # 时间 | 级别 | 内容
_MAX_BYTES = 5 * 1024 * 1024   # 单个日志文件上限 5MB，超出自动轮转
_BACKUP_COUNT = 3              # 最多保留 3 个历史轮转文件


def _make_logger(name: str, filename: str, level: int) -> logging.Logger:
    """创建一个只写文件的独立日志器（幂等：重复导入不会重复挂 handler）"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # 不向 root 传播，防止终端重复打印
    if logger.handlers:        # 已初始化过：直接复用（Streamlit 多次 import 场景）
        return logger
    try:
        LOG_DIR.mkdir(exist_ok=True)   # 首次运行时创建 logs/ 目录
        handler = RotatingFileHandler(LOG_DIR / filename,
                                      maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
                                      encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)
    except OSError:
        # 目录创建/文件打开失败（只读磁盘等）：降级为空日志器，主流程不受影响
        logger.addHandler(logging.NullHandler())
    return logger


# 全局唯一的两个业务/错误日志器（其他模块统一从本模块导入使用）
business_logger = _make_logger("zhixue.business", "business.log", logging.INFO)
error_logger = _make_logger("zhixue.error", "error.log", logging.ERROR)
