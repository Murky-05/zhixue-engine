# -*- coding: utf-8 -*-
"""
utils.storage —— 用户文件物理存储层（多租户物理隔离，零第三方依赖）
====================================================================
职责：把每个用户上传的课件原件保存到项目根目录下的「用户专属目录」：

    项目根/
    ├── data/
    │   ├── 小明/光合作用.pdf        <- 只有"小明"能读写
    │   ├── 小红/光合作用.pdf        <- 与小明的同名文件互不干扰
    │   └── 访客/...

安全约定（多租户隔离的三道防线）：
    1. 目录名防线：user_name 必须通过 validate_user_name() 校验
       （拒绝空名、路径分隔符 / \\、".." 穿越与 Windows 非法字符），
       从源头上保证 data/{user_name}/ 永远是 BASE_DIR 下的一层子目录；
    2. 归属防线：read_original / delete_original 只接受「位于当前用户
       专属目录内」的路径——即使数据库中的 store_path 被篡改指向他人
       目录，也会因归属校验失败而拒绝读写；
    3. 文件名防线：_safe_filename 剥离一切路径成分（浏览器上传的
       original_filename 一般安全，这里做纵深防御）。

与数据库的配合（见 utils/db.py）：
    - files 表新增 store_path 列，记录相对路径 "data/{user}/{filename}"；
    - 删除课件时先做引用计数（count_files_by_store_path），确认没有
      其他课件记录引用同一物理文件后，才调用 delete_original 删除原件。

使用示例：
    from utils import storage
    path = storage.save_original("小明", "光合作用.pdf", raw_bytes)
    data = storage.read_original(path, "小明")       # 返回 bytes
    data = storage.read_original(path, "小红")       # None（跨用户拒绝）
    storage.list_user_files("小明")                  # ["光合作用.pdf"]
"""

import re
import time
from pathlib import Path, PurePosixPath

# 项目根目录（utils/ 的上一级）；data/ 专属存储区的根
ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "data"

# Windows 文件系统的非法字符（含控制字符）——目录名与文件名共用同一套黑名单
_ILLEGAL_FS_CHARS = re.compile(r'[/\\<>:"|?*\x00-\x1f]')


# ---------- 用户名校验（昵称即目录名，必须文件系统安全） ----------
def validate_user_name(user_name):
    """
    校验昵称能否安全地用作存储目录名，返回首尾去空白后的昵称。
    :raises ValueError: 昵称为空 / 含路径分隔符或 Windows 非法字符 /
                        以点号或空格结尾（Windows 保留名怪癖）
    注册（utils/auth.register）与每次落盘前都会调用，双保险。
    """
    name = str(user_name or "").strip()
    if not name:
        raise ValueError("用户名不能为空")
    if len(name) > 64:
        raise ValueError("用户名过长（最多 64 个字符）")
    if _ILLEGAL_FS_CHARS.search(name) or name in (".", ".."):
        raise ValueError('用户名不能包含 \\ / : * ? " < > | 等特殊字符')
    if name.endswith(".") or name.endswith(" "):
        raise ValueError("用户名不能以点号或空格结尾")
    return name


def _safe_filename(filename):
    """
    清洗上传文件名：剥离路径成分、替换非法字符，返回安全的纯文件名。
    （同名文件按约定直接覆盖，见 save_original）
    """
    # Path(...).name 剥离任何目录成分（防 "uploads/../x.pdf" 类构造）
    name = Path(str(filename or "")).name.strip()
    # Windows 非法字符替换为下划线（浏览器一般已拦截，这里做纵深防御）
    name = _ILLEGAL_FS_CHARS.sub("_", name)
    if not name or name in (".", ".."):
        raise ValueError(f"文件名不合法：{filename!r}")
    return name


# ---------- 用户专属目录 ----------
def user_dir(user_name, create=True):
    """
    返回当前用户的专属目录绝对路径（data/{user_name}/）。
    :param create: True 时目录不存在则自动创建（首次上传/落盘用）
    """
    safe_user = validate_user_name(user_name)
    u_dir = BASE_DIR / safe_user
    if create:
        u_dir.mkdir(parents=True, exist_ok=True)
    return u_dir


def _resolve_owned(store_path, user_name):
    """
    把数据库中的 store_path 解析为绝对路径，并强制校验「恰好位于
    当前用户的专属目录一层之内」。三重检查：
      1) store_path 各级不能出现 ".."（拒绝穿越写法）；
      2) 解析后的父目录必须与 data/{user_name}/ 完全一致（防篡改指向他人目录）；
      3) 用户名本身先过 validate_user_name（防昵称夹带路径符号）。
    :return: 校验通过的绝对 Path；不合法/跨用户返回 None
    """
    try:
        safe_user = validate_user_name(user_name)
    except ValueError:
        return None
    if not store_path:
        return None
    parts = PurePosixPath(str(store_path)).parts
    if ".." in parts or "." in parts:
        return None   # 拒绝任何穿越写法
    target = (ROOT / str(store_path)).resolve()
    base = (BASE_DIR / safe_user).resolve()
    if target.parent != base:
        return None   # 不在当前用户专属目录内：拒绝（归属防线）
    return target


# ---------- 增 / 读 / 删 / 列 ----------
# 落盘分块大小（1MB）：分块写入 + 进度回调，让大文件保存有可见进度
SAVE_CHUNK_SIZE = 1024 * 1024


def save_original(user_name, filename, data, progress_cb=None):
    """
    把上传原件保存到当前用户专属目录（同名覆盖），返回 store_path。
    断点续传语义（需求 4）：
      - 分块写入：每写 1MB 触发一次 progress_cb(已写字节, 总字节)，前端据此刷新进度条；
      - 原子落盘：先写 "{文件名}.part" 临时文件，全部写完才重命名为正式文件——
        传输/保存被中断时只会留下 .part 残留（不会被误认为已保存的课件），
        残留可被「系统维护」的孤儿清理回收，上传页检测到 .part 时会提示用户。
    :param progress_cb: 可选回调 progress_cb(written_bytes, total_bytes)
    :raises ValueError: 用户名或文件名不合法（调用方给出友好提示即可）
    """
    safe_user = validate_user_name(user_name)
    safe_name = _safe_filename(filename)
    target = user_dir(safe_user) / safe_name
    part = target.with_name(target.name + ".part")
    total = len(data)
    try:
        written = 0
        with open(part, "wb") as f:
            while written < total:
                chunk = bytes(data[written:written + SAVE_CHUNK_SIZE])
                f.write(chunk)
                written += len(chunk)
                if progress_cb:
                    progress_cb(written, total)
        part.replace(target)   # 原子重命名：只有完整写入的文件才"生效"
    except OSError:
        try:
            part.unlink(missing_ok=True)   # 中断/写盘失败：清掉半截 .part
        except OSError:
            pass
        raise
    if progress_cb:
        progress_cb(total, total)
    return f"data/{safe_user}/{safe_name}"


def read_original(store_path, user_name):
    """
    读取当前用户名下的原件字节。
    :return: bytes；文件不存在 / 路径非法 / 跨用户访问一律返回 None
    """
    target = _resolve_owned(store_path, user_name)
    if target is None:
        return None
    try:
        return target.read_bytes() if target.is_file() else None
    except OSError:
        return None


def delete_original(store_path, user_name):
    """
    删除当前用户名下的原件（调用前应先做数据库引用计数，见 db.count_files_by_store_path）。
    :return: True 已删除；False 文件不存在 / 路径非法 / 跨用户拒绝
    """
    target = _resolve_owned(store_path, user_name)
    if target is None:
        return False
    try:
        if target.is_file():
            target.unlink()
            return True
        return False
    except OSError:
        return False


def list_user_files(user_name):
    """
    列出当前用户专属目录下的全部课件原件（只列一层，子目录忽略；
    排除上传中断残留的 .part 临时文件——它们不是完整课件）。
    其他用户的目录对本函数完全不可见——文件列表隔离的物理保证。
    :return: 文件名列表（按名称排序）；目录不存在返回 []
    """
    try:
        u_dir = user_dir(user_name, create=False)
    except ValueError:
        return []
    if not u_dir.is_dir():
        return []
    return sorted(p.name for p in u_dir.iterdir()
                  if p.is_file() and not p.name.endswith(".part"))


def list_part_files(user_name):
    """列出当前用户目录下的 .part 残留文件（上传被中断的证据，供前端友好提示）"""
    try:
        u_dir = user_dir(user_name, create=False)
    except ValueError:
        return []
    if not u_dir.is_dir():
        return []
    return sorted(p.name for p in u_dir.iterdir() if p.is_file() and p.name.endswith(".part"))


# ---------- 磁盘占用统计（个人中心 / 管理员数据总览） ----------
def dir_size_bytes(user_name):
    """
    统计当前用户专属目录占用的硬盘空间（字节；目录不存在返回 0）。
    只统计本人目录——磁盘占用统计同样遵守多租户隔离。
    """
    try:
        u_dir = user_dir(user_name, create=False)
    except ValueError:
        return 0
    if not u_dir.is_dir():
        return 0
    return sum(p.stat().st_size for p in u_dir.iterdir() if p.is_file())


def total_size_bytes():
    """
    统计 data/ 专属存储区的全部磁盘占用（字节）——
    逐用户目录求和（管理员"数据总览"页展示"磁盘总占用"用）。
    """
    if not BASE_DIR.is_dir():
        return 0
    return sum(
        p.stat().st_size
        for p in BASE_DIR.rglob("*") if p.is_file()
    )


# ---------- 存储管理（管理员"系统维护"页：盘点 / 孤儿清理 / 旧日志清理） ----------
def list_storage_files():
    """
    盘点 data/ 全部用户目录下的文件（含 .part 残留）。
    :return: [(user_name, filename, rel_path, size_bytes)]；只扫一层用户目录
    """
    files = []
    if not BASE_DIR.is_dir():
        return files
    for u_dir in BASE_DIR.iterdir():
        if not u_dir.is_dir():
            continue   # data/ 根下的散落文件不属于任何用户，交由人工处理
        for p in u_dir.iterdir():
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            files.append((u_dir.name, p.name, f"data/{u_dir.name}/{p.name}", size))
    return files


def user_dir_sizes():
    """
    每用户目录的磁盘占用一览（"系统维护"页展示）。
    :return: [(user_name, size_bytes, file_count)]，按占用从大到小排序
    """
    agg = {}
    for user, _name, _rel, size in list_storage_files():
        slot = agg.setdefault(user, [0, 0])
        slot[0] += size
        slot[1] += 1
    return sorted(((u, v[0], v[1]) for u, v in agg.items()), key=lambda x: -x[1])


def scan_orphan_files(referenced_rel_paths):
    """
    盘点"孤儿文件"：磁盘上存在、但 files 表已无任何记录引用的课件原件
    （含上传中断残留的 .part 文件——它们永远不会被数据库引用）。
    :param referenced_rel_paths: 数据库引用的相对路径集合（db.get_referenced_store_paths）
    :return: [(user_name, filename, rel_path, size_bytes)]（只盘点不删除）
    """
    refs = set(referenced_rel_paths or ())
    return [f for f in list_storage_files() if f[2] not in refs]


def cleanup_orphan_files(referenced_rel_paths):
    """
    删除全部孤儿文件（先 scan 再删；被占用等删除失败的文件跳过，下次清理再试）。
    :return: (deleted_count, freed_bytes, deleted_rel_paths)
    """
    deleted, freed, paths = 0, 0, []
    for user, name, rel, size in scan_orphan_files(referenced_rel_paths):
        try:
            (BASE_DIR / user / name).unlink()
            deleted += 1
            freed += size
            paths.append(rel)
        except OSError:
            continue
    return deleted, freed, paths


def list_old_logs(days=7):
    """盘点 logs/ 目录中超过 days 天未更新的旧日志（轮转残留）。
    :return: [(filename, size_bytes, age_days)]（只盘点不删除）"""
    log_dir = ROOT / "logs"
    if not log_dir.is_dir():
        return []
    now = time.time()
    result = []
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        try:
            age_days = (now - p.stat().st_mtime) / 86400
            if age_days > days:
                result.append((p.name, p.stat().st_size, age_days))
        except OSError:
            continue
    return result


def cleanup_old_logs(days=7):
    """
    删除 logs/ 中超过 days 天未更新的旧日志（轮转残留；活跃日志因持续写入不会命中，
    即使被 Windows 文件锁占用也只跳过不报错）。返回 (deleted_count, freed_bytes)。
    """
    log_dir = ROOT / "logs"
    if not log_dir.is_dir():
        return 0, 0
    cutoff = time.time() - days * 86400
    count, freed = 0, 0
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        try:
            if p.stat().st_mtime < cutoff:
                size = p.stat().st_size
                p.unlink()
                count += 1
                freed += size
        except OSError:
            continue   # 活跃日志被系统占用：跳过，下轮清理再试
    return count, freed
