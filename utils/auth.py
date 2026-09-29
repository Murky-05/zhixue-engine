# -*- coding: utf-8 -*-
"""
utils.auth —— 用户认证层（注册 + 加盐哈希密码登录，零第三方依赖）
================================================================
角色约定：
    student  学习端（默认角色）：上传课件 / 问答 / 诊断 / 路径 / 评估
    admin    开发者端：额外可查看全局数据。管理员凭 .env 中的 ADMIN_PASSWORD
             登录（首次登录自动在 users 表创建 admin 账号）

密码安全（纯标准库 hashlib + secrets + hmac）：
    存储格式：pbkdf2_sha256$<迭代次数>$<盐hex>$<哈希hex>
    算法：PBKDF2-HMAC-SHA256，16 字节随机盐，12 万次迭代——
    即使数据库泄露，攻击者也无法反查明文（只能逐用户暴力尝试）。
    校验使用 hmac.compare_digest 恒定时间比较，防时序攻击。

会话约定（与 app.py 的 st.session_state 配合）：
    st.session_state["user_name"]  当前昵称（None=未登录，显示登录页）
    st.session_state["role"]       当前角色（"student" / "admin"）

使用示例：
    from utils.auth import register, login, login_admin, logout
    register("小明", "1234")            # 注册（成功即登录，role=student）
    login("小明", "1234")               # 密码登录（学生/管理员账号通用）
    login_admin("开发者", "xxx")        # 管理员入口（.env 的 ADMIN_PASSWORD）
    logout()                            # 退出：复位身份，回到登录页
"""

import hashlib
import hmac
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

from utils.db import get_user, create_user, set_user_password, touch_last_login, log_event
from utils.storage import validate_user_name   # 昵称即专属存储目录名，注册时校验文件系统安全性

# .env 位于项目根目录（utils/ 的上一级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

VALID_ROLES = ("student", "admin")
MIN_PASSWORD_LEN = 4        # 密码最短位数（本地学习工具，从宽不从严）
PBKDF2_ITERATIONS = 120_000  # PBKDF2 迭代次数：兼顾安全性与登录耗时（<0.1s）


# ================= 密码哈希 =================
def hash_password(pwd):
    """
    生成加盐哈希：PBKDF2-HMAC-SHA256（随机盐，每次调用产生不同密文）。
    :return: "pbkdf2_sha256$120000$<盐hex>$<哈希hex>"（迭代次数存入密文，便于将来升级）
    """
    salt = secrets.token_hex(16)   # 16 字节随机盐（hex 32 字符）
    digest = hashlib.pbkdf2_hmac(
        "sha256", str(pwd).encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(pwd, stored):
    """
    校验明文密码与存储的哈希是否匹配。
    :param stored: hash_password 生成的密文（格式非法/为空一律返回 False）
    """
    if not stored or not pwd:
        return False
    try:
        algo, iterations, salt, digest = str(stored).split("$")
        if algo != "pbkdf2_sha256":
            return False
        calc = hashlib.pbkdf2_hmac(
            "sha256", str(pwd).encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
        # 恒定时间比较：无论从哪一位开始不同，耗时相同，防时序侧信道
        return hmac.compare_digest(calc, digest)
    except (ValueError, TypeError):
        return False


def _session():
    """
    获取 Streamlit 会话状态对象。
    agents/测试等 bare 模式下访问不到会话时返回 None，由调用方兜底处理。
    """
    try:
        import streamlit as st
        return st.session_state
    except Exception:
        return None


def _set_session(user_name, role):
    """把登录身份写入会话（login/register/login_admin 共用）"""
    sess = _session()
    if sess is not None:
        sess["user_name"] = user_name
        sess["role"] = role


def get_current_user():
    """
    读取当前登录用户信息。
    :return: {"user_name": str 或 None, "role": "student"/"admin" 或 None}
             无会话或未登录时 user_name 为 None（app.py 据此显示登录页）
    """
    sess = _session()
    if sess is not None and sess.get("user_name"):
        return {"user_name": sess["user_name"], "role": sess.get("role") or "student"}
    return {"user_name": None, "role": None}


def is_admin():
    """判断当前会话用户是否为开发者（admin）角色"""
    return get_current_user()["role"] == "admin"


def verify_admin_password(password):
    """
    校验管理员密码（与 .env 中的 ADMIN_PASSWORD 比对）。
    :return: True 密码正确；False 错误或 .env 未配置
    """
    expected = os.getenv("ADMIN_PASSWORD")
    if not expected or not password:
        return False
    return str(password) == expected


# ================= 注册 / 登录 / 登出 =================
def register(user_name, password):
    """
    注册新学生账号（成功即自动登录，role=student）。
    :param user_name: 昵称（即登录名，唯一）
    :param password: 密码（>= 4 位）
    :return: {"user_name", "role"} 登录后的身份
    :raises ValueError: 昵称为空 / 密码过短 / 昵称已被注册
    兼容迁移：旧版"昵称即登录"产生的无密码账号，注册同名昵称视为"认领"——
    为其补设密码后正常登录，历史学习记录无缝保留。
    """
    user_name = (user_name or "").strip()
    password = str(password or "")
    if not user_name:
        raise ValueError("昵称不能为空")
    if len(user_name) > 20:
        raise ValueError("昵称最多 20 个字符")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"密码至少 {MIN_PASSWORD_LEN} 位")
    # 昵称同时是专属存储目录名（data/{user_name}/）：必须文件系统安全，
    # 拒绝路径分隔符等特殊字符，从源头防止多租户目录穿越
    try:
        validate_user_name(user_name)
    except ValueError:
        raise ValueError('昵称不能包含 \\ / : * ? " < > | 等特殊字符，也不能以点号或空格结尾')
    # 输入校验（防 XSS/markdown 注入）：新注册昵称只允许中文、字母、数字、
    # 下划线、连字符与空格——从源头杜绝 <script>、markdown 链接等注入载荷
    import re
    if not re.fullmatch(r"[\w\u4e00-\u9fff\- ]+", user_name):
        raise ValueError("昵称只能包含中文、字母、数字、下划线、连字符与空格")

    existing = get_user(user_name)
    if existing is None:
        user = create_user(user_name, hash_password(password), role="student")
    elif existing.get("is_disabled"):
        raise ValueError("该账号已被管理员禁用，如有疑问请联系管理员")
    elif not existing.get("password_hash"):
        # 旧版无密码账号：认领（补设密码），不视为重名冲突
        set_user_password(user_name, hash_password(password))
        user = get_user(user_name)
    else:
        raise ValueError(f"昵称「{user_name}」已被注册，请直接登录或换个昵称")

    _set_session(user_name, "student")
    touch_last_login(user_name)
    try:
        log_event("register", "注册新账号", user_name=user_name, role="student")
    except Exception:
        pass   # 日志失败不影响注册
    return {"user_name": user_name, "role": "student"}


def login(user_name, password):
    """
    密码登录（学生账号通用；admin 账号从此入口登录后仍进入开发者端）。
    :param user_name: 注册时的昵称
    :param password: 注册时设置的密码
    :return: {"user_name", "role"} 登录后的身份
    :raises ValueError: 昵称为空 / 用户未注册 / 密码错误 / 旧账号未认领
    """
    user_name = (user_name or "").strip()
    if not user_name:
        raise ValueError("昵称不能为空")
    user = get_user(user_name)
    if user is None:
        raise ValueError("该昵称尚未注册，请先注册")
    if not user.get("password_hash"):
        raise ValueError("该昵称来自旧版本，请通过「注册」设置密码后使用")
    if user.get("is_disabled"):
        raise ValueError("该账号已被管理员禁用，如有疑问请联系管理员")
    if not verify_password(password, user["password_hash"]):
        raise ValueError("密码错误，请重试")

    _set_session(user_name, user["role"])
    touch_last_login(user_name)
    try:
        log_event("admin_login" if user["role"] == "admin" else "login",
                  "密码登录成功", user_name=user_name, role=user["role"])
    except Exception:
        pass
    return {"user_name": user_name, "role": user["role"]}


def login_admin(user_name, password):
    """
    管理员入口：凭 .env 中的 ADMIN_PASSWORD 登录开发者端。
    首次登录自动在 users 表创建 admin 账号（密码哈希同步记录，便于审计）；
    .env 换密码后仍以 .env 为准校验，数据库哈希仅作存档。
    :raises ValueError: 昵称为空
    :raises PermissionError: 管理密码错误或 .env 未配置 ADMIN_PASSWORD
    """
    user_name = (user_name or "").strip()
    if not user_name:
        raise ValueError("昵称不能为空")
    if not verify_admin_password(password):
        raise PermissionError("管理员密码错误，无法以开发者身份登录")

    # 账号落库（幂等）：首次创建；已存在则补齐密码哈希存档
    user = get_user(user_name)
    if user is None:
        create_user(user_name, hash_password(password), role="admin")
    elif not user.get("password_hash"):
        set_user_password(user_name, hash_password(password))

    _set_session(user_name, "admin")
    touch_last_login(user_name)
    try:
        log_event("admin_login", "管理员登录成功", user_name=user_name, role="admin")
    except Exception:
        pass
    return {"user_name": user_name, "role": "admin"}


def logout():
    """
    退出登录：复位会话身份（main() 据此回到登录页）。
    业务数据（课件/问答/诊断等）的清理由 app.py 的 logout_pending 机制在
    widget 实例化之前完成——此处只处理身份，保持职责单一。
    """
    user = get_current_user()
    sess = _session()
    if sess is not None:
        sess["user_name"] = None
        sess["role"] = None
    try:
        log_event("logout", "退出登录", user_name=user["user_name"] or "未登录",
                  role=user["role"] or "student")
    except Exception:
        pass
    return {"user_name": None, "role": None}
