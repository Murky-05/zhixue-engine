# -*- coding: utf-8 -*-
"""
utils.db —— SQLite 数据库访问层
=================================
集中管理"智学 AI 学习助手"的全部数据库操作（建表 + 增查改删），
app.py 与各 Agent 统一从这里读写，避免 SQL 语句散落各处。

表结构（files 为主表，其余 4 张业务表通过 file_id 外键关联）：
    files      课件文件   id, filename, upload_time, data
                          data = 解析产物 JSON（chunks/知识点候选/图谱），
                          用于"历史状态恢复"：换会话后仍能完整还原课件上下文
    qa         智能问答   id, file_id, question, answer, source_chunks, timestamp
    diagnosis  学习诊断   id, file_id, questions_json, user_answers, score, report, timestamp
    path       学习路径   id, file_id, path_text, timestamp
    eval       学习评估   id, file_id, pre_score, post_score, alg, report, timestamp

多用户约定（多租户隔离）：
    - 所有表带 user_name/role 列；所有业务写入（save_*）必须显式传入当前用户，
      不再提供"访客"隐式默认值——漏传即报错，杜绝数据误归属；
    - 读取按 user_name+role 过滤；None 表示全部用户，仅供开发者端统计使用；
    - get_file_data / delete_file 强制归属校验：学生端传他人 file_id 一律按
      "不存在"处理，防止越权读写。

存储约定：
    - 上传课件的原件字节保存到 data/{user_name}/{原始文件名}（见 utils/storage.py），
      files.store_path 记录相对路径 "data/{user}/{filename}"；
      删除课件时先做引用计数，无其他记录引用同一文件才删物理原件；
    - 所有时间字段统一 "%Y-%m-%d %H:%M:%S" 文本格式；
    - 列表/字典类型的数据（source_chunks、questions、answers、report、data）
      以 JSON 字符串入库（ensure_ascii=False 保留中文）；
    - 外键约束开启，file_id 必须是 files 表中已存在的 id；
    - 老库自动迁移：init_db() 会检测旧表结构并自动补列，历史数据归属"访客"。

使用示例：
    from utils.db import init_db, save_file, save_qa, get_history
    init_db()
    file_id = save_file("光合作用.pdf", data={"chunks": [...]}, user_name="小明")
    save_qa(file_id, "什么是光反应？", "光反应是...", ["chunk_001"], user_name="小明")
    print(get_history(user_name="小明"))
"""

import json
import secrets
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

# 文件日志：业务操作流水 -> logs/business.log（INFO）；异常堆栈 -> logs/error.log（ERROR）
from utils.log_config import business_logger, error_logger

# 数据库文件位置：项目根目录（做成模块常量，便于测试时替换）
DB_PATH = Path(__file__).resolve().parent.parent / "learning_records.db"
# 数据库备份目录：项目根目录 / backup/
BACKUP_DIR = Path(__file__).resolve().parent.parent / "backup"

TIME_FMT = "%Y-%m-%d %H:%M:%S"   # 统一时间格式
DEFAULT_USER = "访客"             # 未输入昵称时的默认用户


# ---------- 基础连接 ----------
def get_conn():
    """
    获取数据库连接：
      - 开启外键约束（SQLite 默认关闭，需每次连接时设置）；
      - row_factory 设为 sqlite3.Row，查询结果可按列名取值。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _now():
    """当前时间的统一格式文本"""
    return datetime.now().strftime(TIME_FMT)


# ---------- 数据库备份与恢复 ----------
def backup_db():
    """
    在线备份数据库：将 learning_records.db 复制到 backup/ 文件夹，时间戳命名。
      - 使用 SQLite 官方 backup API（src.backup(dst)）做在线备份：即使有写入
        并发进行，得到的也是一份原子、一致的完整副本（直接 shutil.copy 正在
        写入的 SQLite 文件可能拷到半截事务，有损坏风险）；
      - 自动创建 backup/ 目录；文件名格式 learning_records_YYYYMMDD_HHMMSS.db。
    :return: 备份文件名（不含目录），如 "learning_records_20260929_120000.db"
    """
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_name = f"learning_records_{stamp}.db"
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(BACKUP_DIR / dst_name)
    try:
        src.backup(dst)   # SQLite 在线备份 API：原子且一致
    finally:
        dst.close()
        src.close()
    business_logger.info(f"数据库已备份 -> backup/{dst_name}")
    return dst_name


def list_db_backups():
    """
    列出 backup/ 目录下全部备份文件名，按时间倒序（最新在前）。
    目录不存在或无备份时返回空列表。
    """
    if not BACKUP_DIR.exists():
        return []
    return sorted((p.name for p in BACKUP_DIR.glob("learning_records_*.db")),
                  reverse=True)


def restore_db(backup_name):
    """
    用指定备份覆盖当前数据库（shutil.copy 整文件覆盖）。
      - 覆盖前先把当前数据库再备份一次（防误恢复的保底快照，文件名带 _pre_restore 标记）；
      - 调用方在恢复完成后应 st.rerun() 重建应用状态。
    :param backup_name: backup/ 目录下的备份文件名
    :return: 恢复前自动创建的"当前库保底备份"文件名
    :raises ValueError: 备份文件不存在
    """
    src = BACKUP_DIR / backup_name
    if not src.exists():
        raise ValueError(f"备份文件不存在：{backup_name}")
    # 恢复前的现状保底快照（独立 pre_restore_ 前缀命名，避免与源备份同秒同名冲突；
    # 不匹配 list_db_backups 的 learning_records_* 通配，不会混入可恢复列表）
    pre_name = f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    cur = sqlite3.connect(DB_PATH)
    snap = sqlite3.connect(BACKUP_DIR / pre_name)
    try:
        cur.backup(snap)
    finally:
        snap.close()
        cur.close()
    shutil.copy(src, DB_PATH)          # 用备份覆盖当前数据库
    business_logger.warning(f"数据库已从 backup/{backup_name} 恢复"
                            f"（恢复前快照：backup/{pre_name}）")
    return pre_name


def _to_json(obj):
    """序列化辅助：list/dict 转 JSON 字符串（保留中文），字符串原样返回"""
    if isinstance(obj, (list, dict)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj) if obj is not None else None


# ---------- 用户（双端分离的用户体系） ----------
def get_user(user_name):
    """按昵称查询用户；不存在返回 None"""
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT id, user_name, role, created_at, password_hash, last_login_at, is_disabled "
            "FROM users WHERE user_name = ?",
            (user_name,),
        ).fetchone()
    return dict(row) if row else None


def create_user(user_name, password_hash, role="student"):
    """
    注册新用户（密码哈希由 auth 层生成后传入，db 层不接触明文）。
    :return: 新用户 dict；昵称已存在抛 ValueError（调用方先做重名检查亦可兜底）
    """
    with closing(get_conn()) as conn:
        try:
            conn.execute(
                "INSERT INTO users (user_name, role, created_at, password_hash) "
                "VALUES (?, ?, ?, ?)",
                (user_name, role, _now(), password_hash),
            )
            conn.commit()
        except Exception as e:
            if "UNIQUE" in str(e):
                raise ValueError(f"昵称「{user_name}」已被注册") from e
            raise
    return get_user(user_name)


def set_user_password(user_name, password_hash):
    """为已有用户设置/更新密码哈希（旧版昵称账号认领、管理员改密时调用）"""
    with closing(get_conn()) as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE user_name = ?",
            (password_hash, user_name),
        )
        conn.commit()


def touch_last_login(user_name):
    """登录成功后刷新最近登录时间（失败不影响登录流程）"""
    try:
        with closing(get_conn()) as conn:
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE user_name = ?",
                (_now(), user_name),
            )
            conn.commit()
    except Exception:
        error_logger.error("touch_last_login 失败（user=%s）", user_name, exc_info=True)
        pass


def upsert_user(user_name, role="student"):
    """
    用户不存在则创建，存在则更新角色（登录时调用）。
    :param role: "student"（学习端）或 "admin"（开发者端，需凭 .env 管理员密码）
    """
    with closing(get_conn()) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (user_name, role, created_at) VALUES (?, ?, ?)",
            (user_name, role, _now()),
        )
        conn.execute("UPDATE users SET role = ? WHERE user_name = ?", (role, user_name))
        conn.commit()
    return get_user(user_name)


def get_all_users():
    """列出全部注册用户（开发者端"用户管理"页），按注册时间升序"""
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, user_name, role, created_at, is_disabled FROM users ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- 账号管控（管理员禁用/启用） ----------
def set_user_disabled(user_name, disabled):
    """
    设置账号禁用状态（管理员"用户管理"页调用）。
    :param disabled: True=禁用（禁止登录，已在会话中的用户下次交互被踢出）；False=启用
    admin 账号不允许禁用（防止把管理员自己锁在门外），此处直接拒绝。
    :return: True 设置成功；False 用户不存在或目标是 admin 账号
    """
    user = get_user(user_name)
    if user is None or user.get("role") == "admin":
        return False
    with closing(get_conn()) as conn:
        conn.execute(
            "UPDATE users SET is_disabled = ? WHERE user_name = ?",
            (1 if disabled else 0, user_name),
        )
        conn.commit()
    return True


def is_user_disabled(user_name):
    """查询账号是否被禁用；用户不存在返回 False（由登录流程另行报"未注册"）"""
    user = get_user(user_name)
    return bool(user and user.get("is_disabled"))


# ---------- 持久登录会话（刷新浏览器后凭 URL 令牌自动恢复登录） ----------
# 原理：Streamlit 的 st.session_state 存活于服务端的 websocket 会话，
# 浏览器一刷新就是全新会话（所有状态清空）。为此登录成功后签发一个随机令牌：
#   令牌写入 sessions 表 + 挂到 URL（?t=xxx）——刷新后 main() 从 URL 取回
#   令牌、查表恢复 user_name/role，实现"刷新不掉线"。
# 安全边界：本地学习工具，令牌明文存库即可；30 天过期，登出/被禁用即删除。
def create_login_session(user_name, role):
    """签发新的持久登录令牌（登录/注册成功后调用），返回令牌字符串。
    顺带惰性清理 30 天前的过期令牌，避免 sessions 表无限膨胀。"""
    token = secrets.token_urlsafe(32)
    with closing(get_conn()) as conn:
        conn.execute(
            "DELETE FROM sessions WHERE created_at < ?",
            ((datetime.now() - timedelta(days=30)).strftime(TIME_FMT),),
        )
        conn.execute(
            "INSERT INTO sessions (token, user_name, role, created_at) VALUES (?, ?, ?, ?)",
            (token, user_name, role, _now()),
        )
        conn.commit()
    return token


def get_login_session(token):
    """按令牌查询登录会话；无效/已删除返回 None。
    :return: {"token", "user_name", "role"} 或 None"""
    if not token:
        return None
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT token, user_name, role FROM sessions WHERE token = ?", (token,)
        ).fetchone()
    return dict(row) if row else None


def delete_login_session(token):
    """删除登录令牌（登出 / 账号被禁用踢下线时调用），令牌立即失效"""
    if not token:
        return
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


# ---------- 建表 ----------
def init_db():
    """
    创建全部表（幂等：表已存在时跳过），并对老库自动迁移补列：
      - users           双端用户体系表（student/admin）
      - files.data      解析产物 JSON（历史恢复用）
      - *.user_name     记录归属用户（本地多用户隔离）
      - *.role          记录归属角色（双端分离：student/admin）
    ALTER 只在列缺失时执行，重复调用无副作用。
    """
    with closing(get_conn()) as conn:
        # ---- 第 1 步：老库迁移必须先行 ----
        # 旧库的表已存在（CREATE TABLE IF NOT EXISTS 会跳过），若先执行
        # 建索引语句 CREATE INDEX ON qa(user_name, ...) 会因旧表缺列直接报错。
        # 因此先补列、再建表/建索引，顺序不能颠倒。
        ROLE_COL = ("role", "TEXT NOT NULL DEFAULT 'student'")
        migrations = {
            "users":     [   # 老库升级：补密码哈希与最近登录时间（旧账号 password_hash 为 NULL）
                ("password_hash", "TEXT"),
                ("last_login_at", "TEXT"),
                ("is_disabled", "INTEGER NOT NULL DEFAULT 0"),   # 账号禁用标记（管理员管控用）
            ],
            "event_log": [],   # 新表：只建不迁
            "sessions": [],    # 新表：持久登录令牌（刷新浏览器后自动恢复登录）
            "files":     [("data", "TEXT"), ("user_name", "TEXT NOT NULL DEFAULT '访客'"), ROLE_COL,
                          ("store_path", "TEXT")],   # 原件物理存储相对路径（data/{user}/{filename}）
            "qa":        [("user_name", "TEXT NOT NULL DEFAULT '访客'"), ROLE_COL,
                          ("session_id", "TEXT"),          # 多会话：所属会话 id（老记录为 NULL）
                          ("session_name", "TEXT"),        # 多会话：会话名称（侧边栏列表展示）
                          ("meta_json", "TEXT")],          # 多会话：恢复气泡用的上下文 JSON
            "diagnosis": [("user_name", "TEXT NOT NULL DEFAULT '访客'"), ROLE_COL],
            "path":      [("user_name", "TEXT NOT NULL DEFAULT '访客'"), ROLE_COL],
            "eval":      [("user_name", "TEXT NOT NULL DEFAULT '访客'"), ROLE_COL],
            "api_usage": [ROLE_COL],
            "tasks":     [],   # 新表：后台任务追踪（解析/诊断/路径的执行状态与错误信息）
        }
        for table, columns in migrations.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not existing:
                continue   # 表还不存在（全新库），交给下面的建表脚本
            for col, decl in columns:
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        conn.commit()

        # ---- 第 2 步：建表 + 建索引（幂等，补列后索引不会缺列） ----
        conn.executescript(
            """
            -- 用户表：双端用户体系（注册+密码登录；student=学习端 / admin=开发者端）
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name     TEXT NOT NULL UNIQUE,     -- 昵称即登录名（唯一）
                role          TEXT NOT NULL DEFAULT 'student',
                created_at    TEXT NOT NULL,
                password_hash TEXT,                      -- 加盐哈希（NULL=旧版昵称账号/未设密码）
                last_login_at TEXT,                      -- 最近一次登录时间
                is_disabled   INTEGER NOT NULL DEFAULT 0 -- 1=已被管理员禁用（禁止登录）
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,        -- 随机登录令牌（挂在 URL ?t= 上）
                user_name  TEXT NOT NULL,           -- 令牌归属用户
                role       TEXT NOT NULL,           -- 令牌归属角色
                created_at TEXT NOT NULL            -- 签发时间（30 天过期，签发时惰性清理）
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name  TEXT,                    -- 发起任务的用户
                task_type  TEXT NOT NULL,           -- 任务类型（parse_pdf / diagnosis_quiz / diagnosis_report / path_generate）
                status     TEXT NOT NULL DEFAULT 'pending',  -- pending/processing/success/failed
                error_msg  TEXT,                    -- 失败原因（成功或进行中为 NULL）
                created_at TEXT NOT NULL,           -- 任务创建时间（即插入 pending 的时刻）
                updated_at TEXT NOT NULL            -- 最近一次状态变更时间
            );

            CREATE TABLE IF NOT EXISTS files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                data        TEXT,                -- 解析产物 JSON（chunks/候选/图谱）
                user_name   TEXT NOT NULL DEFAULT '访客',
                role        TEXT NOT NULL DEFAULT 'student',
                store_path  TEXT                 -- 原件物理存储相对路径（data/{user}/{filename}）
            );

            CREATE TABLE IF NOT EXISTS qa (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id       INTEGER NOT NULL REFERENCES files(id),
                question      TEXT,
                answer        TEXT,
                source_chunks TEXT,               -- JSON 数组：["chunk_001", ...]
                user_name     TEXT NOT NULL DEFAULT '访客',
                role          TEXT NOT NULL DEFAULT 'student',
                timestamp     TEXT NOT NULL,
                session_id    TEXT,               -- 多会话：所属会话 id（NULL = 旧版独立记录）
                session_name  TEXT,               -- 多会话：会话名称（侧边栏列表展示）
                meta_json     TEXT                -- 多会话：恢复气泡用 JSON（contexts/知识点/来源）
            );

            CREATE TABLE IF NOT EXISTS diagnosis (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id        INTEGER NOT NULL REFERENCES files(id),
                questions_json TEXT,               -- 题目列表 JSON
                user_answers   TEXT,               -- 用户答案列表 JSON
                score          INTEGER,            -- 答对题数
                report         TEXT,               -- 诊断报告（JSON/Markdown 文本）
                user_name      TEXT NOT NULL DEFAULT '访客',
                role           TEXT NOT NULL DEFAULT 'student',
                timestamp      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS path (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id    INTEGER NOT NULL REFERENCES files(id),
                path_text  TEXT,                   -- 学习路径（JSON/Markdown 文本）
                user_name  TEXT NOT NULL DEFAULT '访客',
                role       TEXT NOT NULL DEFAULT 'student',
                timestamp  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS eval (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id    INTEGER NOT NULL REFERENCES files(id),
                pre_score  INTEGER,                -- 前测得分
                post_score INTEGER,                -- 后测得分
                alg        REAL,                   -- 归一化学习增益
                report     TEXT,                   -- 评估报告文本（Markdown）
                user_name  TEXT NOT NULL DEFAULT '访客',
                role       TEXT NOT NULL DEFAULT 'student',
                timestamp  TEXT NOT NULL
            );

            -- API 用量表：每次 DeepSeek 调用成功后记录 token 消耗与估算费用（"用量信息"页数据源）
            CREATE TABLE IF NOT EXISTS api_usage (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT NOT NULL,
                user_name         TEXT NOT NULL DEFAULT '访客',
                role              TEXT NOT NULL DEFAULT 'student',
                prompt_tokens     INTEGER,         -- 输入 tokens
                completion_tokens INTEGER,         -- 输出 tokens
                total_tokens      INTEGER,         -- 总 tokens
                estimated_cost    REAL             -- 估算费用（元）：输入1元/M + 输出2元/M
            );

            -- 系统日志表：记录登录/登出/上传/删除等关键事件（开发者端"系统日志"页数据源）
            CREATE TABLE IF NOT EXISTS event_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_name TEXT NOT NULL DEFAULT '访客',
                role      TEXT,                      -- 事件发生时的角色（可空）
                event     TEXT NOT NULL,             -- 事件类型：login/logout/upload/delete_file...
                detail    TEXT                       -- 事件补充说明（可空）
            );

            CREATE INDEX IF NOT EXISTS idx_qa_user     ON qa(user_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_diag_user   ON diagnosis(user_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_eval_file   ON eval(file_id);
            CREATE INDEX IF NOT EXISTS idx_usage_user  ON api_usage(user_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_log_time    ON event_log(timestamp);
            """
        )
        conn.commit()
        # ---- 第 3 步：默认用户"访客"保底存在（首次运行/老库升级后都能对齐） ----
        upsert_user(DEFAULT_USER, "student")


# ---------- 增 ----------
def save_file(filename, data=None, *, user_name, role="student", store_path=None):
    """
    保存课件记录，返回自增 file_id。
    :param data: 解析产物 dict {"chunks": [...], "knowledge_candidates": [...],
                 "graph": {"nodes": [...], "edges": [...]}}（自动转 JSON）
    :param user_name: 归属用户（必传）——多租户约定：所有业务写入必须绑定当前用户
    :param role: 记录归属角色（student/admin）
    :param store_path: 原件物理存储相对路径 "data/{user}/{filename}"
                       （由 utils/storage.save_original 落盘后返回；旧数据/落盘失败为 None）
    """
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO files (filename, upload_time, data, user_name, role, store_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (filename, _now(), _to_json(data), user_name, role, store_path),
        )
        conn.commit()
        return cur.lastrowid


def update_file_data(file_id, data):
    """回填/更新课件的解析产物（如上传时图谱后构建完成的情况）"""
    with closing(get_conn()) as conn:
        conn.execute("UPDATE files SET data = ? WHERE id = ?", (_to_json(data), file_id))
        conn.commit()


def save_qa(file_id, question, answer, source_chunks, *, user_name, role="student",
            session_id=None, session_name=None, meta=None):
    """
    保存问答记录，返回记录 id。
    :param source_chunks: 来源片段编号列表（如 ["chunk_001"]）或 JSON 文本
    :param user_name: 归属用户（必传）
    :param role: 记录归属角色（student/admin）
    :param session_id: 所属会话 id（多会话管理；None 表示独立记录）
    :param session_name: 所属会话名称（侧边栏会话列表展示用）
    :param meta: 恢复对话气泡用的附加信息 dict（contexts/knowledge_points/source_page/source_snippet）
    """
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO qa (file_id, question, answer, source_chunks, user_name, role, timestamp, "
            "session_id, session_name, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (file_id, question, answer, _to_json(source_chunks), user_name, role, _now(),
             session_id, session_name, _to_json(meta)),
        )
        conn.commit()
        return cur.lastrowid


# ---------- 会话管理（智能问答多会话） ----------
def get_qa_session_meta(session_id, user_name, role="student"):
    """
    查询问答会话元信息（带归属校验）：刷新后凭 URL 中的 session_id 恢复会话用。
    :return: {"session_name", "file_id"}；会话不存在 / 不属于该用户返回 None
             （file_id 用于联动恢复课件上下文；旧版无 file_id 的记录返回 None 值字段）
    """
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT session_name, file_id FROM qa "
            "WHERE session_id = ? AND user_name = ? AND role = ? "
            "ORDER BY id DESC LIMIT 1",
            (session_id, user_name, role),
        ).fetchone()
    if row is None:
        return None
    return {"session_name": row["session_name"], "file_id": row["file_id"]}


def list_qa_sessions(user_name, role="student", file_id=None):
    """
    列出某用户在某课件下的问答会话（按最后提问时间倒序）。
    会话名取该组最新一条记录的 session_name（自动命名后会同步全组，此处为双保险）。
    :return: [{"session_id", "session_name", "count", "last_time"}]
             session_id 为 NULL 的旧版独立记录不纳入会话列表
    """
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT q1.session_id, "
            "(SELECT q2.session_name FROM qa q2 WHERE q2.session_id=q1.session_id "
            " AND q2.user_name=? AND q2.role=? ORDER BY q2.id DESC LIMIT 1) AS session_name, "
            "COUNT(*) AS count, MAX(q1.timestamp) AS last_time "
            "FROM qa q1 WHERE q1.user_name=? AND q1.role=? AND q1.file_id=? "
            "AND q1.session_id IS NOT NULL "
            "GROUP BY q1.session_id ORDER BY last_time DESC, MAX(q1.id) DESC",
            (user_name, role, user_name, role, file_id),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_qas(session_id, user_name, role="student"):
    """
    取某会话的全部问答记录（按时间正序，恢复对话气泡用）。
    安全：user_name+role 双重过滤，只取当前用户自己的记录。
    """
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT question, answer, meta_json, session_name, timestamp FROM qa "
            "WHERE session_id=? AND user_name=? AND role=? ORDER BY id ASC",
            (session_id, user_name, role),
        ).fetchall()
    return [dict(r) for r in rows]


def rename_qa_session(session_id, session_name, user_name, role="student"):
    """重命名会话（只改该用户自己的会话，双过滤防越权）"""
    with closing(get_conn()) as conn:
        conn.execute(
            "UPDATE qa SET session_name=? WHERE session_id=? AND user_name=? AND role=?",
            (session_name, session_id, user_name, role),
        )
        conn.commit()


def delete_qa_session(session_id, user_name, role="student"):
    """删除会话：级联删除该会话的全部问答记录（单事务保证一致性）"""
    with closing(get_conn()) as conn:
        conn.execute(
            "DELETE FROM qa WHERE session_id=? AND user_name=? AND role=?",
            (session_id, user_name, role),
        )
        conn.commit()


def save_diagnosis(file_id, questions, answers, score, report, *, user_name, role="student"):
    """
    保存诊断记录，返回记录 id。
    :param questions: 题目列表（自动转 JSON）
    :param answers:   用户答案列表（自动转 JSON）
    :param score:     答对题数
    :param report:    诊断报告（dict/Markdown 文本均可，dict 自动转 JSON）
    :param user_name: 归属用户（必传）
    :param role:      记录归属角色（student/admin）
    """
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO diagnosis (file_id, questions_json, user_answers, score, report, user_name, role, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (file_id, _to_json(questions), _to_json(answers), int(score), _to_json(report),
             user_name, role, _now()),
        )
        conn.commit()
        return cur.lastrowid


def save_path(file_id, path_text, *, user_name, role="student"):
    """保存学习路径记录（list/dict 自动转 JSON），返回记录 id（user_name 必传）"""
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO path (file_id, path_text, user_name, role, timestamp) VALUES (?, ?, ?, ?, ?)",
            (file_id, _to_json(path_text), user_name, role, _now()),
        )
        conn.commit()
        return cur.lastrowid


def save_eval(file_id, pre_score, post_score, alg, report=None, *, user_name, role="student"):
    """
    保存学习评估记录（前测/后测得分、ALG 增益与报告文本），返回记录 id。
    :param report: 评估报告文本（Markdown 字符串；dict 会自动转 JSON）
    :param user_name: 归属用户（必传）
    :param role:   记录归属角色（student/admin）
    """
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO eval (file_id, pre_score, post_score, alg, report, user_name, role, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (file_id, int(pre_score), int(post_score), float(alg), _to_json(report),
             user_name, role, _now()),
        )
        conn.commit()
        return cur.lastrowid


def save_api_usage(prompt_tokens, completion_tokens, total_tokens, estimated_cost,
                   *, user_name, role="student"):
    """
    保存一条 API 用量记录（BaseAgent.chat 每次成功调用后写入），返回记录 id。
    :param user_name: 归属用户（必传）
    :param role: 记录归属角色（student/admin）
    """
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO api_usage (timestamp, user_name, role, prompt_tokens, completion_tokens, "
            "total_tokens, estimated_cost) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), user_name, role, int(prompt_tokens), int(completion_tokens),
             int(total_tokens), float(estimated_cost)),
        )
        conn.commit()
        return cur.lastrowid


# ---------- 查 ----------
def _filters(alias=None, user_name=None, role=None):
    """
    构造统一的 WHERE 过滤条件（双端分离的核心查询约定）：
      - user_name 有值 -> AND user_name = ?
      - role 有值      -> AND role = ?
      两者都为 None 时返回空条件（查全部）。
    :param alias: 表别名（JOIN 查询时限定列，避免同名列歧义）
    :return: (where_sql, params) 二元组
    """
    col = f"{alias}." if alias else ""
    conds, params = [], []
    if user_name:
        conds.append(f"{col}user_name = ?")
        params.append(user_name)
    if role:
        conds.append(f"{col}role = ?")
        params.append(role)
    return (f"WHERE {' AND '.join(conds)}" if conds else "", tuple(params))


def get_api_usage_summary(user_name=None, role=None):
    """
    汇总 API 用量（"用量信息"页指标卡数据源）。
    :param user_name: 只统计该用户；None 表示全部用户
    :param role: 只统计该角色（student/admin）；None 表示全部角色
    """
    where, params = _filters(user_name=user_name, role=role)
    with closing(get_conn()) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS requests, "
            f"COALESCE(SUM(prompt_tokens), 0)     AS prompt_tokens, "
            f"COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
            f"COALESCE(SUM(total_tokens), 0)      AS total_tokens, "
            f"COALESCE(SUM(estimated_cost), 0)    AS total_cost "
            f"FROM api_usage {where}", params,
        ).fetchone()
    return dict(row)


def get_api_usage_daily(user_name=None, role=None):
    """
    按日期聚合的每日消费金额（"用量信息"页折线图数据源）。
    :return: dict {date_str: cost}（按日期升序；无记录返回空 dict）
    """
    where, params = _filters(user_name=user_name, role=role)
    with closing(get_conn()) as conn:
        rows = conn.execute(
            f"SELECT substr(timestamp, 1, 10) AS day, "
            f"COALESCE(SUM(estimated_cost), 0) AS cost "
            f"FROM api_usage {where} GROUP BY day ORDER BY day ASC", params,
        ).fetchall()
    return {r["day"]: r["cost"] for r in rows}


def get_api_usage_by_user():
    """按用户聚合的全平台 API 用量（开发者端"全平台用量"页，按消费降序）"""
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT user_name, role, COUNT(*) AS requests, "
            "COALESCE(SUM(total_tokens), 0) AS tokens, "
            "COALESCE(SUM(estimated_cost), 0) AS cost "
            "FROM api_usage GROUP BY user_name, role ORDER BY cost DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_api_usage(limit=100):
    """
    最近的 API 调用明细（开发者端"系统日志"页）。
    :return: [{"timestamp", "user_name", "role", "prompt_tokens",
               "completion_tokens", "total_tokens", "estimated_cost"}]（时间倒序）
    """
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT timestamp, user_name, role, prompt_tokens, completion_tokens, "
            "total_tokens, estimated_cost FROM api_usage ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_daily_activity(days=7):
    """
    近 N 天全平台活跃度（每天的学习行为 + API 调用总事件数）。
    统计范围：上传/问答/诊断/路径/评估/API调用 六类事件按天计数。
    :return: dict {date_str: count}（仅包含有事件的日期，按日期升序）
    """
    cutoff = (datetime.now() - timedelta(days=int(days) - 1)).strftime("%Y-%m-%d")
    sql_union = " UNION ALL ".join(
        f"SELECT substr({col}, 1, 10) AS d FROM {table} WHERE substr({col}, 1, 10) >= ?"
        for table, col in (
            ("files", "upload_time"), ("qa", "timestamp"), ("diagnosis", "timestamp"),
            ("path", "timestamp"), ("eval", "timestamp"), ("api_usage", "timestamp"),
        )
    )
    with closing(get_conn()) as conn:
        rows = conn.execute(
            f"SELECT d, COUNT(*) AS n FROM ({sql_union}) GROUP BY d ORDER BY d ASC",
            (cutoff,) * 6,
        ).fetchall()
    return {r["d"]: r["n"] for r in rows}


def get_registration_daily():
    """
    按日期统计全平台新增注册用户数（开发者端"系统监控"页折线图数据源）。
    :return: dict {date_str: count}（仅包含有注册的日期，按日期升序）
    """
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n FROM users "
            "GROUP BY d ORDER BY d ASC"
        ).fetchall()
    return {r["d"]: r["n"] for r in rows}


def get_user_stats():
    """
    全平台用户统计（开发者端"用户管理"页）：
    每个用户的角色、注册时间、上传文件数、API 消费、最后活跃时间。
    最后活跃 = 该用户在 event_log / api_usage / qa 三表中最近一次记录的时间。
    :return: [{"user_name", "role", "created_at", "files", "cost", "last_active", "is_disabled"}]
    """
    with closing(get_conn()) as conn:
        rows = conn.execute(
            """
            SELECT u.user_name, u.role, u.created_at, u.is_disabled,
                   (SELECT COUNT(*) FROM files f
                     WHERE f.user_name = u.user_name AND f.role = u.role) AS files,
                   COALESCE((SELECT SUM(estimated_cost) FROM api_usage a
                              WHERE a.user_name = u.user_name AND a.role = u.role), 0) AS cost,
                   -- 三参数 MAX(a,b,c) 是标量取大；切勿再包一层单参数 MAX()
                   -- （单参数 MAX 是聚合函数，无 GROUP BY 时会把全表折叠成 1 行！）
                   MAX(
                       COALESCE((SELECT MAX(timestamp) FROM event_log e
                                 WHERE e.user_name = u.user_name), ''),
                       COALESCE((SELECT MAX(timestamp) FROM api_usage a2
                                 WHERE a2.user_name = u.user_name AND a2.role = u.role), ''),
                       COALESCE((SELECT MAX(timestamp) FROM qa q
                                 WHERE q.user_name = u.user_name AND q.role = u.role), '')
                   ) AS last_active
            FROM users u ORDER BY u.id ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_admin_overview():
    """
    全平台数据总览（开发者端"数据总览"页）：
    各表记录数 + 全局 API 消耗汇总。
    """
    with closing(get_conn()) as conn:
        counts = {}
        for table in ("users", "files", "qa", "diagnosis", "path", "eval"):
            counts[table] = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        usage = conn.execute(
            "SELECT COUNT(*) AS requests, "
            "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
            "COALESCE(SUM(estimated_cost), 0) AS total_cost FROM api_usage"
        ).fetchone()
    counts.update({"requests": usage["requests"], "total_tokens": usage["total_tokens"],
                   "total_cost": usage["total_cost"]})
    return counts


# ---------- 后台任务追踪（"任务后台"与"监控告警"数据源） ----------
TASK_STATUSES = ("pending", "processing", "success", "failed")


def create_task(task_type, user_name, role=None):
    """
    登记一个后台任务（status=pending），返回任务 id。
    生命周期：pending（登记）-> processing（开始执行）-> success / failed（终态）。
    """
    now = _now()
    with closing(get_conn()) as conn:
        cur = conn.execute(
            "INSERT INTO tasks (user_name, task_type, status, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?)",
            (user_name, task_type, now, now),
        )
        conn.commit()
        business_logger.info("[任务] 创建 #%s %s (user=%s)", cur.lastrowid, task_type, user_name)
        return cur.lastrowid


def update_task_status(task_id, status, error_msg=None):
    """
    更新任务状态并刷新 updated_at；status='failed' 时必须带 error_msg。
    状态非法或任务不存在时返回 False（调用方无需中断主流程）。
    """
    if status not in TASK_STATUSES:
        return False
    try:
        with closing(get_conn()) as conn:
            cur = conn.execute(
                "UPDATE tasks SET status = ?, error_msg = COALESCE(?, error_msg), "
                "updated_at = ? WHERE id = ?",
                (status, error_msg, _now(), task_id),
            )
            conn.commit()
            if status in ("success", "failed"):
                business_logger.info("[任务] #%s -> %s %s",
                                     task_id, status, error_msg or "")
            return cur.rowcount > 0
    except sqlite3.Error:
        error_logger.error("update_task_status 失败（task_id=%s）", task_id, exc_info=True)
        return False


def get_task_counts_by_status():
    """各状态任务计数 {"pending": n, "processing": n, "success": n, "failed": n}（缺失状态补 0）"""
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
    counts = {s: 0 for s in TASK_STATUSES}
    for r in rows:
        counts[r["status"]] = r["n"]
    return counts


def get_failed_tasks(limit=50):
    """
    最近的失败任务列表（管理员"任务监控"页数据源）。
    :return: [{"id", "user_name", "task_type", "error_msg", "created_at", "updated_at"}]
    """
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, user_name, task_type, status, error_msg, created_at, updated_at "
            "FROM tasks WHERE status = 'failed' ORDER BY updated_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_task_daily_stats(days=7):
    """
    近 N 天每日任务成败计数（管理员"任务监控"趋势图数据源）。
    :return: [{"day": "2026-09-29", "success": n, "failed": n}]（按日期升序）
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime(TIME_FMT)
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT substr(updated_at, 1, 10) AS day, "
            "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success, "
            "SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed "
            "FROM tasks WHERE status IN ('success', 'failed') AND updated_at >= ? "
            "GROUP BY day ORDER BY day",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_task_counts_since(hours=24):
    """最近 N 小时的任务计数 {"total": n, "failed": n}（失败激增告警用）"""
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime(TIME_FMT)
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed "
            "FROM tasks WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()
    return {"total": row["total"] or 0, "failed": row["failed"] or 0}


def get_stuck_tasks(minutes=15):
    """
    疑似卡死的任务：status 仍是 processing 但超过 minutes 分钟没有状态更新。
    （"系统告警"页用——长时间无更新通常意味着执行线程异常退出）
    """
    cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime(TIME_FMT)
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, user_name, task_type, created_at, updated_at "
            "FROM tasks WHERE status = 'processing' AND updated_at < ? "
            "ORDER BY updated_at ASC LIMIT 20",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_referenced_store_paths():
    """
    files 表引用的全部物理路径集合（"系统维护"孤儿清理的比对基准）。
    store_path 非空的记录直接采用；旧记录（store_path 为 NULL）按约定路径
    "data/{user}/{filename}" 推导——保证孤儿清理绝不误删仍被旧课件记录引用的原件。
    :return: set[str]，元素形如 "data/{user}/{filename}"
    """
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT user_name, filename, store_path FROM files").fetchall()
    refs = set()
    for r in rows:
        if r["store_path"]:
            refs.add(r["store_path"])
        else:
            refs.add(f"data/{r['user_name']}/{r['filename']}")
    return refs


# ---------- 系统日志 ----------
def log_event(event, detail=None, user_name=DEFAULT_USER, role=None):
    """写入一条系统事件日志（登录/登出/上传/删除等关键动作）。
    双通道：event_log 表（前端系统日志页可见）+ logs/business.log（文件流水，可追溯）。"""
    business_logger.info("[%s] %s (user=%s, role=%s)", event, detail or "-", user_name, role)
    with closing(get_conn()) as conn:
        conn.execute(
            "INSERT INTO event_log (timestamp, user_name, role, event, detail) VALUES (?, ?, ?, ?, ?)",
            (_now(), user_name, role, event, detail),
        )
        conn.commit()


def get_recent_logs(limit=100, event=None):
    """
    读取最近的系统日志（开发者端"系统日志"页）。
    :param limit: 返回条数上限（默认 100）
    :param event: 按事件类型过滤（None 表示全部）
    :return: [{"timestamp", "user_name", "role", "event", "detail"}]（时间倒序）
    """
    cond, qp = ("WHERE event = ?", [event]) if event else ("", [])
    with closing(get_conn()) as conn:
        rows = conn.execute(
            f"SELECT timestamp, user_name, role, event, detail FROM event_log {cond} "
            f"ORDER BY id DESC LIMIT ?", (*qp, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_file_data(file_id, *, user_name, role):
    """
    读取课件的解析产物（历史状态恢复的核心），强制归属校验：
    只返回 user_name+role 都匹配的记录——传入他人的 file_id 一律按
    "不存在"处理（返回 None），防止跨用户串号恢复他人课件。
    :param user_name: 当前用户（必传）
    :param role:      当前角色（必传，与 user_name 双重过滤）
    :return: {"chunks", "knowledge_candidates", "graph"} dict；不存在或归属不符返回 None
    """
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT data FROM files WHERE id = ? AND user_name = ? AND role = ?",
            (file_id, user_name, role),
        ).fetchone()
    if row is None or not row["data"]:
        return None
    try:
        data = json.loads(row["data"])
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        error_logger.error("get_file_data 解析产物损坏（file_id=%s, user=%s）", file_id, user_name,
                           exc_info=True)
        return None


def list_files(user_name=None, role=None):
    """
    列出课件文件（供"文件管理"区展示），按上传时间倒序。
    :param user_name: 只看该用户的文件；None 表示全部用户（仅开发者端）
    :param role: 只看该角色的文件（student/admin）；None 表示全部角色
    :return: [{"id", "filename", "upload_time", "user_name", "store_path",
               "qa", "diagnosis", "path", "eval"}]（后四项为各类记录数）
    """
    where, params = _filters(user_name=user_name, role=role)
    sql = "SELECT id, filename, upload_time, user_name, store_path FROM files " + where
    sql += " ORDER BY id DESC"
    with closing(get_conn()) as conn:
        files = [dict(r) for r in conn.execute(sql, params)]
        # 逐文件统计各类记录数（记录量小，逐表 GROUP BY 再合并）
        for f in files:
            counts = {}
            for table in ("qa", "diagnosis", "path", "eval"):
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE file_id = ?", (f["id"],)
                ).fetchone()
                counts[table] = row["n"]
            f.update(counts)
    return files


def get_history(user_name=None, role=None):
    """
    查询学习历史（5 张表联查，按时间倒序合并）。
    :param user_name: 只返回该用户的记录；None 表示全部用户
    :param role: 只返回该角色的记录（student/admin）；None 表示全部角色
    :return: 统一格式的记录列表：
        [{"type": "file"/"qa"/"diagnosis"/"path"/"eval",
          "file_id": int, "filename": str, "time": str, "detail": dict}]
    """
    history = []
    # 业务表与 files 表同有 user_name/role 列，JOIN 查询必须用表别名限定，避免歧义
    def _cond(alias=None):
        return _filters(alias=alias, user_name=user_name, role=role)
    with closing(get_conn()) as conn:
        # 课件上传
        where, params = _cond()
        for r in conn.execute(f"SELECT id, filename, upload_time FROM files {where}", params):
            history.append({
                "type": "file", "file_id": r["id"], "filename": r["filename"],
                "time": r["upload_time"], "detail": {},
            })
        # 问答（LEFT JOIN 关联文件名，file 被删时文件名置空）
        where, params = _cond("q")
        for r in conn.execute(
            f"""SELECT q.file_id, q.question, q.answer, q.source_chunks, q.timestamp, f.filename
                FROM qa q LEFT JOIN files f ON q.file_id = f.id {where}""",
            params,
        ):
            history.append({
                "type": "qa", "file_id": r["file_id"], "filename": r["filename"] or "（已删除）",
                "time": r["timestamp"],
                "detail": {"question": r["question"], "answer": r["answer"],
                           "source_chunks": r["source_chunks"]},
            })
        # 诊断
        where, params = _cond("d")
        for r in conn.execute(
            f"""SELECT d.file_id, d.questions_json, d.user_answers, d.score, d.report, d.timestamp, f.filename
                FROM diagnosis d LEFT JOIN files f ON d.file_id = f.id {where}""",
            params,
        ):
            history.append({
                "type": "diagnosis", "file_id": r["file_id"], "filename": r["filename"] or "（已删除）",
                "time": r["timestamp"],
                "detail": {"score": r["score"], "questions": r["questions_json"],
                           "answers": r["user_answers"], "report": r["report"]},
            })
        # 学习路径
        where, params = _cond("p")
        for r in conn.execute(
            f"""SELECT p.file_id, p.path_text, p.timestamp, f.filename
                FROM path p LEFT JOIN files f ON p.file_id = f.id {where}""",
            params,
        ):
            history.append({
                "type": "path", "file_id": r["file_id"], "filename": r["filename"] or "（已删除）",
                "time": r["timestamp"], "detail": {"path_text": r["path_text"]},
            })
        # 学习评估
        where, params = _cond("e")
        for r in conn.execute(
            f"""SELECT e.file_id, e.pre_score, e.post_score, e.alg, e.report, e.timestamp, f.filename
                FROM eval e LEFT JOIN files f ON e.file_id = f.id {where}""",
            params,
        ):
            history.append({
                "type": "eval", "file_id": r["file_id"], "filename": r["filename"] or "（已删除）",
                "time": r["timestamp"],
                "detail": {"pre_score": r["pre_score"], "post_score": r["post_score"],
                           "alg": r["alg"], "report": r["report"]},
            })

    # 按时间倒序（同一时刻的记录按类型稳定排序，保证输出可预期）
    history.sort(key=lambda x: (x["time"], x["type"]), reverse=True)
    return history


def get_history_grouped(user_name=None, role=None, max_files=5):
    """
    按课件分组的学习历史（侧边栏"按课件归档的项目管理"展示的数据源）。
    在 get_history() 平铺结果之上二次分组，不重复写 SQL：
      - 只有 qa/diagnosis/path/eval 四类学习活动的课件才入选（纯上传事件不算）；
      - 组内记录沿用 get_history() 的统一格式（type/file_id/filename/time/detail），
        可直接作为 restore_history 回调的入参，点击即恢复；
      - 组内保持时间倒序（首条即最新一条，用于侧边栏摘要展示）。
    :param user_name: 只返回该用户的记录；None 表示全部用户
    :param role:      只返回该角色的记录；None 表示全部角色
    :param max_files: 最多返回最近活动的几个课件（None 表示不限制）
    :return: 分组列表（按最近活动时间倒序）：
        [{"file_id": int, "filename": str, "latest_time": str,
          "qa": [...], "diagnosis": [...], "path": [...], "eval": [...]}, ...]
    """
    groups = {}   # file_id -> 分组 dict（课件被删除时 filename 会显示"（已删除）"）
    for r in get_history(user_name=user_name, role=role):
        if r["type"] == "file" or r.get("file_id") is None:
            continue   # 纯上传事件不属于 4 类学习活动，跳过
        fid = r["file_id"]
        g = groups.get(fid)
        if g is None:   # 首次遇到该课件：以最新一条记录初始化分组
            g = {"file_id": fid, "filename": r["filename"], "latest_time": r["time"],
                 "qa": [], "diagnosis": [], "path": [], "eval": []}
            groups[fid] = g
        g[r["type"]].append(r)   # get_history 已倒序，append 后组内天然保持倒序
    # 课件之间按最近活动时间倒序，再截取最近 N 个课件
    result = sorted(groups.values(), key=lambda g: g["latest_time"], reverse=True)
    return result[:max_files] if max_files is not None else result


# ---------- 删 ----------
def delete_file(file_id, *, user_name, role):
    """
    删除课件及其全部关联记录（级联删除），强制归属校验：
    只能删除 user_name+role 都匹配的课件——传入他人的 file_id 一律返回 False，
    且子表（qa/diagnosis/path/eval）删除时同样带归属条件，保证"只删自己的记录"。
    :param user_name: 当前用户（必传）
    :param role:      当前角色（必传）
    :return: True 删除成功；False 文件不存在或不属于当前用户
    事务保护：5 张表在同一事务中删除，任一步失败整体回滚，不会留下孤儿记录。
    """
    with closing(get_conn()) as conn:
        try:
            cur = conn.execute(
                "SELECT id FROM files WHERE id = ? AND user_name = ? AND role = ?",
                (file_id, user_name, role),
            )
            if cur.fetchone() is None:
                return False   # 文件不存在或不属于当前用户
            # 先删业务子表，最后删 files 主表（外键约束下顺序不可颠倒）
            for table in ("qa", "diagnosis", "path", "eval"):
                conn.execute(
                    f"DELETE FROM {table} WHERE file_id = ? AND user_name = ? AND role = ?",
                    (file_id, user_name, role),
                )
            conn.execute(
                "DELETE FROM files WHERE id = ? AND user_name = ? AND role = ?",
                (file_id, user_name, role),
            )
            conn.commit()
            return True
        except sqlite3.Error:
            error_logger.error("delete_file 失败（file_id=%s, user=%s）", file_id, user_name,
                               exc_info=True)
            conn.rollback()
            raise


def count_files_by_store_path(store_path):
    """
    统计引用同一物理文件的课件记录数（删除课件后决定是否删除原件用）。
    用法：delete_file 成功后调用——返回 0 说明已无其他记录引用该文件，
    可安全调用 storage.delete_original 删除物理原件；>0 说明同用户还
    有其他课件记录指向同一文件（同名重复上传），原件必须保留。
    """
    if not store_path:
        return 0
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE store_path = ?", (store_path,)
        ).fetchone()
    return row["n"]
