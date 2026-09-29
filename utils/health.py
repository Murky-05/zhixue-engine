# -*- coding: utf-8 -*-
"""
utils.health —— 系统存活体检
=============================
管理员后台「系统监控」页的真实探测数据源，对四大关键依赖逐一"体检"：

  1. SQLite 数据库连通性   -> 真实执行 SELECT 1
  2. DeepSeek API 连通性   -> 真实发一个 1-token 请求（最小成本探活，10 秒超时）
  3. 磁盘剩余空间          -> shutil.disk_usage（系统盘 + 应用所在盘）
  4. 错误日志统计          -> 数 logs/error.log 中昨天/今天的 ERROR 记录条数

设计要点：
  - 纯函数、零 Streamlit 依赖：只返回 (是否正常, 展示文案) 或纯数据 dict，
    UI 渲染全部交给 app.py——本模块可被脚本 / AppTest 单独调用验证；
  - API 探测由前端「重新体检」按钮或首次进页显式触发，结果缓存在
    session_state，绝不在每次 rerun 时自动重发（避免刷新页面就烧 token）；
  - 每项探测各自 try-except 兜底：单项失败不影响其余项出结果，函数永不抛异常；
  - 体检自身的失败【不写】error.log——否则失败探运会污染第 4 项的报错统计。

使用示例：
    from utils.health import run_health_check
    hc = run_health_check()
    print(hc["db"])      # (True, "数据库正常（SELECT 1 通过）")
    print(hc["api"])     # (True, "API 连通（deepseek-chat，耗时 0.83s）")
    print(hc["disk"])    # [("系统盘 C:", 62.3, 180.5), ...]
    print(hc["errors"])  # {"yesterday": 0, "today": 2}
"""

import os
import shutil
import time
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)

# 与 base_agent / auth 相同的 .env 加载约定（模块导入即生效，幂等）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 项目根目录：数据库、logs/ 都在这里；也是"应用所在盘"的探测目标
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

# API 探测超时（秒）：体检要快，10 秒连不上即判不连通
API_TIMEOUT = 10

# 系统盘探测目标：Windows 用 C:，Linux/Docker 容器用根分区
SYS_DISK = "C:/" if os.name == "nt" else "/"

# HTTP 状态码 -> 人话提示（DeepSeek 常见错误；与 base_agent.ERROR_MESSAGES 风格互补，
# 这里面向"管理员排障"，保留原始状态码方便对照官方文档）
HTTP_HINTS = {
    401: "认证失败（API Key 无效或已删除）",
    402: "余额不足（请到 DeepSeek 平台充值）",
    404: "模型不存在（检查模型名）",
    422: "请求参数错误",
    429: "请求过于频繁（触发限流）",
}


def check_database():
    """
    探测 1：SQLite 数据库连通性——真实执行 SELECT 1。
    db 层延迟导入（函数内 import），避免 utils.health 在模块加载期
    就被 utils.db 反向依赖造成的循环导入风险。
    :return: (是否正常, 展示文案)
    """
    try:
        from utils.db import get_conn
        with closing(get_conn()) as conn:
            conn.execute("SELECT 1").fetchone()
        return True, "数据库正常（SELECT 1 通过）"
    except Exception as e:   # 连接失败 / 文件锁死 / 磁盘故障等
        return False, f"数据库异常：{type(e).__name__}: {e}"


def check_deepseek_api():
    """
    探测 2：DeepSeek API 连通性——真实发一个 1-token 请求（最小成本探活）。
      - 直接用 OpenAI 兼容 SDK（与 base_agent 相同的 Key / base_url 约定）；
      - 不走 BaseAgent 的重试网关：体检要快、要真实（重试反而掩盖瞬时故障）；
      - max_tokens=1：只要服务端正常受理并返回，即视为连通。
    :return: (是否正常, 展示文案)；文案包含具体错误码（如 HTTP 402）方便排障
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return False, "未配置 DEEPSEEK_API_KEY（请在项目根目录 .env 中配置）"

    t0 = time.perf_counter()
    try:
        client = OpenAI(api_key=api_key,
                        base_url="https://api.deepseek.com",
                        timeout=API_TIMEOUT)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,   # 只要 1 个 token：最小成本的真实请求
        )
        cost = time.perf_counter() - t0
        return True, f"API 连通（{resp.model}，耗时 {cost:.2f}s）"
    except APITimeoutError:
        return False, f"连接超时（>{API_TIMEOUT}s），网络不通或服务不可达"
    except APIConnectionError:
        return False, "网络连接失败（无法访问 api.deepseek.com）"
    except APIStatusError as e:   # 服务端返回了 HTTP 错误码（401/402/429/5xx...）
        hint = HTTP_HINTS.get(e.status_code, "服务端错误")
        return False, f"HTTP {e.status_code}：{hint}"
    except Exception as e:        # 兜底：SDK 版本差异等未知异常
        return False, f"未知异常：{type(e).__name__}: {e}"


def check_disk():
    """
    探测 3：磁盘剩余空间——shutil.disk_usage 真实读取。
      - 系统盘（Windows=C:，Linux=/）+ 应用所在盘（项目根目录）各一行；
      - Docker 部署时两项通常相同（容器内项目与系统同分区），属正常现象。
    :return: [(盘标签, 已用百分比, 剩余GB), ...]；取不到的盘对应值为 None
    """
    items = []
    for label, target in (("系统盘", SYS_DISK), ("应用盘", ROOT)):
        try:
            total, _, free = shutil.disk_usage(target)
            used_pct = (total - free) / total * 100 if total else None
            items.append((label, used_pct, free / 1024 ** 3 if total else None))
        except OSError:   # 盘符不存在 / 权限不足（如容器内无 C: 盘）
            items.append((label, None, None))
    return items


def count_error_log():
    """
    探测 4：错误日志统计——读取 logs/error.log，数昨天 / 今天的报错条数。
      - log_config 的行格式为 "2026-09-29 21:36:30,738 [ERROR] ..."，
        每条记录只有首行带时间戳，按行首日期前缀计数即等于报错条数
        （多行堆栈的后续行不带日期前缀，不会重复计）；
      - 文件不存在（从未报过错）按 0 处理；只统计当前文件，轮转历史
        （error.log.1 等）不计——昨天的完整数字以日志页为准，这里看趋势。
    :return: (昨日报错条数, 今日报错条数)
    """
    y_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    t_date = datetime.now().strftime("%Y-%m-%d")
    y_cnt = t_cnt = 0
    path = LOG_DIR / "error.log"
    if path.exists():
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith(y_date):
                        y_cnt += 1
                    elif line.startswith(t_date):
                        t_cnt += 1
        except OSError:   # 文件被占用等：按 0 处理，不影响其余体检项
            pass
    return y_cnt, t_cnt


def run_health_check():
    """
    执行全套存活体检（4 项全部真实探测），返回 UI 直接消费的结果字典：
        {
          "checked_at": "YYYY-MM-DD HH:MM:SS",   # 本次体检时间
          "db":   (ok, detail),                   # 探测 1 结果
          "api":  (ok, detail),                   # 探测 2 结果
          "disk": [(label, used_pct, free_gb)],   # 探测 3 结果
          "errors": {"yesterday": n, "today": n}, # 探测 4 结果
        }
    """
    y_cnt, t_cnt = count_error_log()
    return {
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db": check_database(),
        "api": check_deepseek_api(),
        "disk": check_disk(),
        "errors": {"yesterday": y_cnt, "today": t_cnt},
    }
