# -*- coding: utf-8 -*-
"""迁移脚本闭环实测（用后即删）"""
import os
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"e:\智学")
sys.path.insert(0, r"e:\智学")

PASS, FAIL = "✅ PASS", "❌ FAIL"
from migrate import run_migrations, _get_user_version
from utils.db import DB_PATH

conn = sqlite3.connect(DB_PATH)
before = _get_user_version(conn)
conn.close()
print(f"迁移前 user_version = {before}")

v = run_migrations()
print(f"run_migrations() -> user_version = {v}")
print(f"{'✅ PASS' if v == 1 else '❌ FAIL'}  首次运行：user_version 推进到 1")

conn = sqlite3.connect(DB_PATH)
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
conn.close()
expected = {"files", "qa", "diagnosis", "path", "eval", "users", "sessions", "tasks", "event_log"}
print(f"{'✅ PASS' if expected <= set(tables) else '❌ FAIL'}  基础表齐全: {sorted(tables)}")

v2 = run_migrations()
print(f"{'✅ PASS' if v2 == 1 else '❌ FAIL'}  幂等重跑：user_version 保持 1（实际 {v2}）")
