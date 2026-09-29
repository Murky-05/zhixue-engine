# -*- coding: utf-8 -*-
"""
migrate.py —— 数据库版本化迁移（基于 SQLite PRAGMA user_version）
=================================================================
数据库结构版本化的统一入口：app.py 启动时自动调用 run_migrations()，
按版本号顺序执行未应用的迁移，并同步推进 user_version。

工作原理：
    - PRAGMA user_version 是 SQLite 内置的整型版本槽（0 ~ 2147483647），
      随数据库文件一起持久化，天然与应用版本解耦；
    - MIGRATIONS 字典登记每个版本号对应的迁移动作（函数或 SQL 脚本）；
    - run_migrations() 读取当前 user_version，只执行"当前版本之后"的迁移，
      每执行一个就把 user_version 推进到该版本（迁移与版本推进在同一连接，
      天然原子）；
    - 迁移动作必须幂等或仅在首次执行：user_version 一旦到位永不重跑。

新增迁移的方法：
    1. 在 MIGRATIONS 中追加下一个版本号（如 2: _migration_2_add_xxx）；
    2. 迁移函数里写 ALTER TABLE / CREATE TABLE 等真实变更；
    3. 修改 db.py 中 init_db() 的建表逻辑，保证新库直接建出最新结构。

使用示例：
    from migrate import run_migrations
    run_migrations()   # app.py 启动时自动调用
"""

import sqlite3

from utils.db import DB_PATH, business_logger


def _migration_1_create_tables():
    """
    迁移 1：创建全部基础表（files/qa/diagnosis/path/eval/users/sessions/...）。
    复用 utils.db.init_db() 的幂等建表逻辑（CREATE TABLE IF NOT EXISTS），
    避免同一份建表 SQL 在两个文件中重复维护。
    """
    from utils.db import init_db   # 局部导入：避免模块加载期的循环依赖
    init_db()


# 版本号 -> 迁移动作（函数或可 executescript 的 SQL 字符串）
MIGRATIONS = {
    1: _migration_1_create_tables,
}


def _get_user_version(conn: sqlite3.Connection) -> int:
    """读取数据库当前的 user_version"""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    """推进 user_version 到指定版本（与迁移在同一连接内，天然原子）"""
    conn.execute(f"PRAGMA user_version = {int(version)}")


def run_migrations() -> int:
    """
    执行全部未应用的数据库迁移，返回迁移后的 user_version。
      - 无迁移需要执行时直接返回当前版本（幂等，可重复调用）；
      - 每个迁移执行后立即提交并推进 user_version——中途中断时，
        重启后会从"已完成的下一个版本"继续，不会重复执行已应用的迁移。
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        current = _get_user_version(conn)
        latest = max(MIGRATIONS) if MIGRATIONS else current
        if current >= latest:
            conn.close()
            return current

        for version in range(current + 1, latest + 1):
            step = MIGRATIONS[version]
            if callable(step):
                step()   # 迁移函数自行连接数据库并提交
            else:
                conn.executescript(step)   # SQL 脚本形式
            _set_user_version(conn, version)
            conn.commit()
            business_logger.info(f"数据库迁移完成：user_version -> {version}")

        return _get_user_version(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    # 命令行直接运行：手动触发迁移（部署脚本/运维场景）
    v = run_migrations()
    print(f"数据库迁移完成，当前 user_version = {v}")
