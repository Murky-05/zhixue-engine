# -*- coding: utf-8 -*-
"""
智学 AI 学习助手（agents 架构版）
=================================
Streamlit 主程序：只负责页面交互与流程编排，所有智能能力由 agents/ 目录下的
8 个 Agent 提供：

    ParserAgent     多格式课件解析（PDF/Word/TXT/MD）-> 文本块 + 知识点候选（上传页）
    IndexAgent      本地 TF-IDF 检索索引                  （上传页构建）
    RetrieverAgent  混合检索（向量+图谱）-> 最相关片段与知识点 （问答页）
    OntologyAgent   知识图谱构建（知识点 + 依赖关系）       （上传页 / 图谱页）
    TutorAgent      引导式答疑（片段+知识点上下文，标注来源） （问答页）
    DiagnosisAgent  图谱选点出题 + 错因诊断（JSON 报告）    （诊断页）
    PathAgent       先修依赖拓扑排序学习路径（纯本地算法）   （路径页）
    EvalAgent       前测-后测学习增益评估（ALG，结果入库）   （评估页）
    api_usage 表    每次调用成功后自动记录 token 消耗与估算费用（用量信息页）

运行方式：
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import concurrent.futures
import html
import json
import os
import threading
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from agents import (
    DiagnosisAgent,
    EncryptedPDFError,
    EvalAgent,
    IndexAgent,
    OntologyAgent,
    ParseError,
    ParserAgent,
    PathAgent,
    RetrieverAgent,
    ScannedPDFError,
    TopicPlannerAgent,
    TutorAgent,
)
# 数据库访问层：全部学习记录的读写统一走 utils/db.py
from utils.db import (
    delete_file,
    delete_qa_session,
    get_admin_overview,
    get_api_usage_by_user,
    get_api_usage_daily,
    get_api_usage_summary,
    get_daily_activity,
    get_file_data,
    get_history,
    get_history_grouped,
    get_recent_api_usage,
    get_recent_logs,
    get_registration_daily,
    get_session_qas,
    get_user,
    get_user_stats,
    init_db,
    is_user_disabled,
    list_files,
    list_qa_sessions,
    log_event,
    rename_qa_session,
    save_diagnosis,
    save_eval,
    save_file,
    save_path,
    save_qa,
    count_files_by_store_path,
    set_user_disabled,
    create_login_session,
    delete_login_session,
    get_login_session,
    update_file_data,
    # 后台任务追踪（解析/诊断/路径的执行状态与错误信息）
    create_task,
    update_task_status,
    get_task_counts_by_status,
    get_failed_tasks,
    get_task_daily_stats,
    # 数据库备份与恢复（管理员系统维护页）
    backup_db,
    list_db_backups,
    restore_db,
    get_task_counts_since,
    get_stuck_tasks,
    get_referenced_store_paths,
    get_qa_session_meta,
)
# 用户文件物理存储层：上传原件保存到 data/{当前用户}/，多租户物理隔离
from utils import storage
# 图谱可视化：networkx 图 -> pyvis 交互式 HTML 字符串
from utils.graph_viz import build_graph_html
# 评估可视化：plotly 掌握度雷达图 + 零依赖 SVG 报告卡（导出图片用）
from utils.viz import bar_chart_data, radar_fig, radar_png_bytes, radar_report_svg
# 用户认证层：登录/注册/登出（SQLite + 加盐哈希，密码不落明文）
from utils.auth import register, login, login_admin, logout
# 系统存活体检：DB SELECT 1 / DeepSeek 1-token 探活 / 磁盘 / 错误日志统计（管理员监控页用）
from utils.health import run_health_check
# 文件日志：报错堆栈 -> logs/error.log（业务流水经 db.log_event 已写入 business.log）
from utils.log_config import error_logger
# API 网关统一错误码与文案映射（前端展示文案的唯一来源）
from agents.base_agent import ERR_TIMEOUT, ERR_UNKNOWN, ERROR_MESSAGES

# ========== 页面基础配置（必须是第一个 st 命令） ==========
# layout="wide"：登录页需要 100vh 左右分栏全屏布局；
# 登录后页面由全局 CSS 的 .block-container max-width 恢复居中限宽（视觉同 centered）
st.set_page_config(
    page_title="智学引擎 · AI 学习助手",
    layout="wide",
)

# 轻量美化：统一字体与细节（接近原生 App 的观感）+ 平板响应式适配 + 登录页基础设施样式
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
        }
        h1 { letter-spacing: 0.5px; }
        .stTextArea textarea { border-radius: 10px; }

        /* ===== 隐藏 Streamlit 默认菜单 / footer / 顶栏（四重保险，彻底清除顶栏） ===== */
        header, [data-testid="stHeader"], .stApp > header,
        #MainMenu, footer {
            display: block !important; height: 0 !important;
            background: transparent !important; visibility: hidden !important;
            overflow: visible !important;
        }
        /* 侧边栏收起后的展开按钮（stExpandSidebarButton）位于 header 内部——
           上方规则把 header 隐藏后按钮随之不可见，用户收起侧边栏就再也打不开。
           display:none 无法被子级恢复，故 header 改用 visibility:hidden 隐藏，
           这里单独恢复按钮的可见性（收起态左上角仍可点开展开）；
           header 自身 height:0 + 透明 + overflow:visible，顶栏依旧不占位不显眼 */
        header [data-testid="stExpandSidebarButton"],
        [data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        /* ===== 品牌配色：苹果青 + 奶酪色 =====
           全局背景 = 左上奶酪色 → 右下浅苹果青的柔和线性渐变（登录页与内页一体）；
           文字主色 #2C3E50 / 次要 #7F8C8D */
        .stApp {
            background: linear-gradient(135deg, #FBF1D7 0%, #F6F3DC 45%, #E8F3E0 100%) fixed;
            color: #2C3E50;
        }

        /* ===== 清除 CSS 注入块的占位空隙 =====
           st.markdown 注入的 <style> 虽不可见，但其外层 stElementContainer 仍留在
           flex 布局流中，每个贡献一个 gap（顶部白条根源 = 2 × 16px）。
           两步法（该内核不支持嵌套 :has，故用"先全藏 + 按 id 恢复"）：
           ① 含 <style> 的注入块容器整体隐藏；② 品牌块（#brand-pane）恢复显示，
           其 specificity 更高（id 参与 :has 计算），覆盖第 ① 条 */
        [data-testid="stElementContainer"]:has(style) {
            display: none !important;
        }
        [data-testid="stElementContainer"]:has(#brand-main) {
            display: flex !important;
        }
        [data-testid="stMarkdownContainer"] p, .stMarkdown, .stApp label {
            color: #2C3E50;
        }
        [data-testid="stMarkdownContainer"] small, [data-testid="stCaptionContainer"],
        [data-testid="stCheckbox"] label { color: #7F8C8D; }

        /* wide 布局下恢复内页居中限宽（视觉等价于原 centered 模式） */
        .block-container { max-width: 736px; margin: 0 auto; }

        /* ===== 侧边栏排版（Vercel/DeepSeek 式导航：大字号、宽松间距） ===== */
        /* 功能导航（st.radio）菜单项：15px 正文级字号 + 10px 上下内边距，消除拥挤感 */
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            font-size: 15px !important;
            padding: 10px 0 !important;
        }
        /* 选中的菜单项加粗（input:checked 位于选中项 label 内） */
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            font-weight: 700 !important;
        }
        /* 侧边栏分区标题（"最近学习记录"等）：14px 加粗，与正文拉开层级 */
        .sb-sec-title {
            font-size: 14px !important;
            font-weight: 700 !important;
            color: #2C3E50 !important;
            margin: 0 0 4px 0 !important;
        }
        /* 侧边栏普通说明文字统一 13px（比默认略大，保持可读） */
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            font-size: 13px !important;
        }

        /* ===== 侧边栏会话行（[3,1,1] 行式布局）：名称加粗 + 极小灰色操作按钮 ===== */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
            white-space: nowrap !important;
        }
        /* 第一列（会话名称按钮）：加粗左对齐，长名省略号，不被挤扁 */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button {
            font-weight: 700 !important;
            text-align: left !important;
            justify-content: flex-start !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button p {
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
            max-width: 100%;
        }
        /* 当前会话（primary）：覆写为浅绿底深绿字，替代全站绿底白字的厚重感 */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button[data-testid="stBaseButton-primary"] {
            background-color: #EAF4E2 !important;
            border-color: #EAF4E2 !important;
            color: #4A7A2C !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button[data-testid="stBaseButton-primary"]:hover {
            background-color: #DFF0D2 !important;
            border-color: #DFF0D2 !important;
            color: #4A7A2C !important;
        }
        /* 第二、三列（改名/删除）：无框灰字小按钮；固定 2.5rem（40px）与名称按钮
           严格等高（名称钮 = 16px 字号 + 默认内边距实测 40px），flex 居中竖向对齐 */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:not(:first-child) button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #7F8C8D !important;
            font-size: 13px !important;
            padding: 0 4px !important;
            height: 2.5rem !important;
            min-height: 2.5rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:not(:first-child) button:hover {
            background: rgba(44, 62, 80, 0.06) !important;
            color: #2C3E50 !important;
        }

        /* ===== 主按钮品牌色：苹果青 #73AE52，悬浮加深 #629A44（全站统一） ===== */
        button[data-testid="stBaseButton-primary"] {
            background-color: #73AE52 !important;
            border: 1px solid #73AE52 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            transition: background-color .15s ease, border-color .15s ease;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #629A44 !important;
            border-color: #629A44 !important;
        }

        /* ===== 首页 2×2 功能卡矩阵：等高 / 统一内边距 / 按钮规格 =====
           卡片视觉容器实测为 stVerticalBlock（直接子是 stElementContainer）；
           锚点 = 卡内首元素 st.html 注入的 span#home-card-N（st.markdown 会
           重写 id 不可用）。:has 直接子链精确匹配第 3 级，不会命中外层块 */
        div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stHtml"] > span[id^="home-card-"]) {
            height: 100% !important;      /* 拉满 wrapper 高：同行两卡随 flex stretch 等高 */
            min-height: 180px !important;
            padding: 24px !important;
            border-radius: 12px !important;
            display: flex !important;
            flex-direction: column !important;
        }
        /* 等高拉伸链补全：height:100% 在中间层断裂（中间层高度=内容高），
           外层块与 stLayoutWrapper 也必须拉满，卡片 100% 才有确定参照 */
        div[data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"] [id^="home-card-"]) {
            height: 100% !important;
        }
        div[data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"] [id^="home-card-"]) > [data-testid="stLayoutWrapper"] {
            height: 100% !important;
        }
        /* 锚点 span（0 高）的外层容器隐藏，不占卡片内部 flex 空间 */
        [data-testid="stElementContainer"]:has(> [data-testid="stHtml"] > span[id^="home-card-"]) {
            display: none !important;
        }
        /* 卡内按钮：统一 40px 高、撑满列宽（width="stretch"）并贴卡片底部，
           长短描述不影响按钮位置 —— 等高矩阵的视觉关键 */
        div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stHtml"] > span[id^="home-card-"]) button {
            height: 40px !important;
            min-height: 40px !important;
        }
        div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stHtml"] > span[id^="home-card-"]) > [data-testid="stElementContainer"]:has(button) {
            margin-top: auto !important;
        }

        /* ===== 响应式适配（平板 768px-1024px，需求 3） ===== */
        @media (min-width: 769px) and (max-width: 1024px) {
            /* 多列卡片自动降列：容器允许换行，列宽改为弹性两列（3列→2列、4列→2+2）
               注意排除侧边栏：侧边栏内的行式布局（会话行 [3,1,1] 等）必须保持
               原生比例单行，46% 强制列宽会让多列垂直堆叠、按钮错位 */
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
            section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
            [data-testid="stHorizontalBlock"]:not(section[data-testid="stSidebar"] *) > div[data-testid="stColumn"] {
                flex: 1 1 46% !important;
                min-width: 46% !important;
                width: 46% !important;
            }
            /* 侧边栏收窄保持可折叠（原生折叠钮仍可用），内容不溢出 */
            section[data-testid="stSidebar"] { width: 230px !important; min-width: 230px !important; }
            /* 防溢出：长词/长代码/宽表格允许换行或横向滚动 */
            [data-testid="stMarkdownContainer"] * { word-break: break-word; }
            [data-testid="stDataFrame"], .stMarkdown pre { overflow-x: auto; }
            [data-testid="stMarkdownContainer"] pre { white-space: pre-wrap; }
        }
        /* 手机（≤768px）：Streamlit 原生侧边栏自动折叠为抽屉，这里只做防溢出 */
        @media (max-width: 768px) {
            [data-testid="stMarkdownContainer"] * { word-break: break-word; }
            [data-testid="stDataFrame"] { overflow-x: auto; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== 全局配置 ==========
# 记录类型 -> 中文名称（侧边栏历史记录展示用，与 utils/db.py 的 get_history 类型对应）
TYPE_LABELS = {
    "file": "上传",
    "qa": "问答",
    "diagnosis": "诊断",
    "path": "路径",
    "eval": "评估",
}

# ---- Agent 运行配置（开发者端"Agent 配置"页修改，存 st.session_state，执行时读取） ----
# 开关为 False 时对应功能入口给出"管理员已停用"提示，不发起调用；
# 参数项作为学习端界面的默认值 / 检索过滤条件，在功能执行瞬间实时读取。
DEFAULT_AGENT_CONFIG = {
    "parser_enabled": True,        # 解析 Agent：文本切块 + AI 知识点抽取
    "planner_enabled": True,       # 自主规划 Agent：无课件输入主题生成图谱与路径
    "ontology_enabled": True,      # 图谱 Agent：AI 构建知识图谱（含解析第 4 步/重建/增量）
    "tutor_enabled": True,         # 问答 Tutor Agent：引导式回答
    "diagnosis_enabled": True,     # 诊断 Agent：AI 出题 + 诊断报告
    "path_enabled": True,          # 路径 Agent：本地拓扑排序生成学习路径
    "eval_enabled": True,          # 评估 Agent：前后测出题 + AI 增益报告
    "diagnosis_n_questions": 3,    # 诊断出题数量（学习端滑块默认值，可选 3/5/10）
    "retrieval_min_score": 0.10,   # 检索置信度阈值：低于该混合相关度的片段不作为回答依据
}


def get_agent_config():
    """
    读取 Agent 配置（开发者端"Agent 配置"页写入 st.session_state.agent_config）。
    未配置或缺项时自动补默认值，保证学习端在任何情况下都能安全读取。
    """
    cfg = st.session_state.setdefault("agent_config", {})
    for k, v in DEFAULT_AGENT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


def agent_disabled_msg(feature):
    """统一的"功能已被管理员停用"提示（配置入口：开发者端 Agent 配置页）。
    双通道反馈：页面警告 + 右下角 Toast（全局状态反馈机制）。"""
    st.warning(f"管理员已在「Agent 配置」页停用{feature}，如需使用请联系开发者开启。")
    st.toast(f"已拦截：{feature}已被停用")


# ========== 小工具 ==========
def ai_fail_hint(agent, fallback="操作失败，请稍后重试。"):
    """
    Agent 调用失败时的用户端统一提示（全局异常处理约定）：
      - 用户端只显示友好文案（如"AI服务暂时不可用"），绝不出现
        余额 / 充值 / DeepSeek 等技术细节字眼；
      - 具体错误码与完整堆栈已由 BaseAgent 写入 logs/error.log 与运行终端。
    双通道反馈：页面红色错误条 + 右下角红色 Toast（全局状态反馈机制）。
    :param agent: 刚调用失败的 Agent 实例（读取其 last_error 分类信息）
    :param fallback: Agent 未提供分类信息时的兜底文案
    """
    msg = getattr(agent, "last_error", None) or {}
    text = msg.get("message") or fallback
    st.error(f"{text}")
    st.toast(text)


def safe_callback(func):
    """
    回调安全包装：st.button 的 on_click 回调中若抛出任何异常，
    Streamlit 会在页面上渲染红色 Traceback。这里统一拦截：
      - 用户端：设置一条友好提示（显示一次即清除），页面不崩溃；
      - 开发者端：完整堆栈打印到终端。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(f"[ERROR callback] {func.__name__} failed")
            error_logger.error("on_click 回调异常：%s", func.__name__, exc_info=True)
            st.session_state["history_warning"] = "操作未能完成，请刷新页面后重试。"
    return wrapper


# ========== Agent 调用限时降级包装（需求 3：AI 执行超时触发降级提示） ==========
# 默认限时 30 秒（问答类短回复）；生成类长任务（出题/图谱/评估报告）在各调用点
# 显式放宽到 90 秒——DeepSeek 生成长 JSON 的真实耗时常超过 30 秒，一刀切会误伤。
AGENT_CALL_TIMEOUT = 30
# 超时降级的用户端文案（与 base_agent 的安全文案约定一致，不含技术细节）
AGENT_TIMEOUT_MSG = "AI 响应超时，本次操作已自动跳过，请稍后重试"


def _run_with_session_ctx(fn):
    """
    把 fn 包一层：在子线程中执行前，绑定当前脚本的 Streamlit 会话上下文。
    （Agent 内部会读 st.session_state 记录用量归属；不绑定的话子线程读不到会话，
    用量会被误记成"访客"。绑定后 _record_usage / log_event 归属正确。）
    bare 模式（测试脚本）取不到上下文时按原样执行。
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    def _target():
        if ctx is not None:
            try:
                add_script_run_ctx(threading.current_thread(), ctx)
            except Exception:
                pass   # 绑定失败只影响用量归属，不影响调用本身
        return fn()

    return _target


def call_agent_limited(fn, agent=None, timeout=None, spinner_text="AI 正在思考中..."):
    """
    Agent 调用的统一"限时 + 降级"包装（所有 AI 调用点经此执行）：
      1. 在独立子线程中执行 fn()（绑定会话上下文），脚本线程限时等待；
      2. 超过 timeout 秒未返回 -> 不再等待，立即返回 None 并给 agent.last_error
         写入超时文案，调用方据此走"局部降级"分支（页面绝不卡死/崩溃）；
         底层请求若稍后自行完成，其用量仍会正常入库，不影响计费统计；
      3. fn() 意外抛出的异常同样被捕获转为 last_error（双保险——
         正常情况下 base_agent.chat() 内部已兜底，理论上不会抛到这里）。
    :param fn: 无参可调用对象（用 lambda 把 agent 方法与参数包进来）
    :param agent: Agent 实例（超时/异常时写 last_error；传 None 则只返回 None）
    :param timeout: 限时秒数，默认 AGENT_CALL_TIMEOUT（30s）
    :param spinner_text: 执行期间显示的 st.spinner 文案；传 None 不显示
    :return: fn() 的返回值；超时/异常返回 None
    """
    if timeout is None:
        timeout = AGENT_CALL_TIMEOUT
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-call")
    try:
        fut = ex.submit(_run_with_session_ctx(fn))
        try:
            if spinner_text:
                with st.spinner(spinner_text):
                    return fut.result(timeout=timeout)
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"[ERROR agent-timeout] Agent call exceeded {timeout}s -> degrade")
            if agent is not None:
                agent.last_error = {"code": ERR_TIMEOUT, "message": AGENT_TIMEOUT_MSG}
            return None
        except Exception as e:
            # 兜底：调用包装器之外的未知异常也绝不让 Traceback 渲染到页面
            print(f"[ERROR agent-call] {type(e).__name__}: {e}")
            error_logger.error("Agent 调用包装器捕获未知异常", exc_info=True)
            if agent is not None:
                agent.last_error = {"code": ERR_UNKNOWN, "message": ERROR_MESSAGES[ERR_UNKNOWN]}
            return None
    finally:
        # 不等待遗留线程（wait=False）：超时后脚本线程立即返回，底层请求自行了断
        ex.shutdown(wait=False)


# ========== AI 请求轻量限流（需求 2：单用户每分钟最多 5 次 AI 请求） ==========
AI_RATE_LIMIT = 5     # 窗口内允许的最大 AI 请求次数
AI_RATE_WINDOW = 60   # 窗口长度（秒）


def ai_rate_limit_check():
    """
    轻量级 AI 请求限流：单用户每分钟最多发起 AI_RATE_LIMIT 次 AI 请求。
    实现（零依赖，纯 st.session_state）：
      - session["ai_call_times"] 记录最近每次 AI 请求的时间戳；
      - 每次检查先丢弃窗口（60s）外的时间点，再判断窗口内次数是否超限；
      - 未超限：记录本次时间戳并返回 True（调用方继续执行 AI 请求）；
      - 超限：显示友好提示"请求过于频繁，请休息一分钟再试"并返回 False，
        调用方直接 return 短路本次请求（不消耗 API）。
    说明：计数存在会话里，刷新浏览器或重新登录会计数归零——
    对"防手滑/防脚本滥用"足够；若未来需要跨会话精确配额，应升级到数据库计数。
    :return: True=放行（已记录本次请求）；False=超限（提示已显示，调用方短路）
    """
    now = time.time()
    times = [t for t in (st.session_state.get("ai_call_times") or [])
             if now - t < AI_RATE_WINDOW]   # 只保留窗口内的时间点
    if len(times) >= AI_RATE_LIMIT:
        msg = f"请求过于频繁，请休息一分钟再试（每分钟最多 {AI_RATE_LIMIT} 次 AI 请求）"
        st.warning(msg)
        st.toast("请求过于频繁，本次操作已拦截")
        st.session_state["ai_call_times"] = times
        return False
    times.append(now)
    st.session_state["ai_call_times"] = times
    return True


# ========== 未上传课件的统一空状态引导（需求 3：带插图卡片 + 跳转按钮） ==========
def _goto_upload():
    """跳转到上传解析页（on_click 回调：先于下一轮 widget 实例化执行，可安全写 page 键）"""
    st.session_state.page = "上传解析"


def _goto_help():
    """跳转到帮助中心页（on_click 回调：先于下一轮 widget 实例化执行，可安全写 page 键）"""
    st.session_state.page = "帮助中心"


def _sync_sid_param():
    """把当前问答会话 id 同步到地址栏 query 参数（sid）；无会话时移除该参数"""
    sid = st.session_state.get("qa_session_id")
    if sid:
        st.query_params["sid"] = str(sid)
    else:
        st.query_params.pop("sid", None)


# 内联 SVG 插图：云上传 + 上传箭头 + 课件堆叠（Apple 风格浅色扁平，零外部资源）
_SVG_UPLOAD = """
<svg width="128" height="100" viewBox="0 0 128 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="64" cy="46" r="38" fill="#f0f6ff"/>
  <path d="M38 44a12 12 0 0 1 23-5 10 10 0 0 1 19 3 8 8 0 0 1-2 16H40a8 8 0 0 1-2-14z" fill="#dbe9ff"/>
  <rect x="60" y="34" width="5" height="24" rx="2.5" fill="#3478f6"/>
  <path d="M62.5 24l11 13h-22z" fill="#3478f6"/>
  <rect x="34" y="70" width="60" height="9" rx="4.5" fill="#d7dade"/>
  <rect x="42" y="84" width="44" height="9" rx="4.5" fill="#e6e8ec"/>
</svg>
"""


def _empty_card(title, desc, svg=_SVG_UPLOAD):
    """空状态插图卡片（自定义 HTML）：居中插图 + 标题 + 描述，禁止白屏。
    title/desc 一律 html.escape 转义——防 XSS 约定：进入 unsafe 渲染前必须转义。
    注意：HTML 必须单行拼接（无换行缩进、无内部空行）——st.markdown 的
    Markdown 解析在空行处断开 HTML 块，且缩进 4+ 空格的行会被当作代码块，
    两者都会导致 HTML 源码以文本形式泄露到页面上。"""
    title = html.escape(str(title))
    desc = html.escape(str(desc))
    st.markdown(
        f'<div style="text-align:center; padding:34px 18px; margin:6px 0;'
        f' border:1px solid rgba(0,0,0,.06); border-radius:18px;'
        f' background:linear-gradient(180deg,#ffffff,#f7f8fa);">'
        f'{svg}'
        f'<p style="font-size:1.06rem; font-weight:600; color:#1d1d1f;'
        f' margin:10px 0 4px;">{title}</p>'
        f'<p style="font-size:.92rem; color:#86868b; margin:0;">{desc}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def empty_doc_guide():
    """
    未上传课件时的统一空状态引导（问答/图谱/诊断/路径/评估五个页面共用）：
    插图卡片文案 + "去上传"跳转按钮，禁止白屏或裸提示。
    """
    _empty_card(
        "您还没有上传课件，请先前往【上传解析】",
        "支持 PDF / Word / TXT / Markdown，解析后即可使用问答、诊断、图谱等全部功能",
    )
    st.button("前往上传解析", type="primary", on_click=_goto_upload,
              use_container_width=True)


def ensure_file_id():
    """
    确保当前课件在 files 表中有对应记录，返回 file_id。
    兼容旧会话：课件解析于迁移前（无 file_id）时，补录一条 files 记录。
    """
    if st.session_state.get("file_id") is None and st.session_state.doc["filename"]:
        st.session_state.file_id = save_file(
            st.session_state.doc["filename"],
            user_name=st.session_state.user_name, role=st.session_state.role,
        )
    return st.session_state.file_id


def render_history_detail(record):
    """侧边栏展开单条历史记录的完整内容（点击 expander 即可回溯）"""
    d = record["detail"]
    rtype = record["type"]
    if rtype == "qa":
        st.markdown(f"**问题：** {d.get('question', '')}")
        st.markdown(d.get("answer") or "")
    elif rtype == "diagnosis":
        st.markdown(f"**得分：{d.get('score', '?')}**")
        try:
            # 报告以 JSON 字符串入库，解析后复用诊断报告渲染
            st.markdown(render_diagnosis_md(json.loads(d.get("report") or "")))
        except json.JSONDecodeError:
            st.markdown(d.get("report") or "")   # 旧版 Markdown 文本直接显示
    elif rtype == "path":
        try:
            for s in json.loads(d.get("path_text") or "[]"):
                st.markdown(f"**{s['order']}. {s['knowledge_point']}**")
                st.caption(s.get("reason", ""))
        except json.JSONDecodeError:
            st.markdown(d.get("path_text") or "")
    elif rtype == "eval":
        c1, c2, c3 = st.columns(3)
        c1.metric("前测", d.get("pre_score", 0))
        c2.metric("后测", d.get("post_score", 0))
        c3.metric("ALG", f"{d.get('alg', 0):.2f}")
        st.markdown(d.get("report") or "")   # eval 表 report 列：完整评估报告文本
    elif rtype == "file":
        st.caption("上传了该课件")


def reset_eval_state():
    """清空学习评估（前测/后测）相关的全部会话状态"""
    for k in ("evaluation", "pre_quiz", "pre_score", "post_quiz", "post_score"):
        st.session_state[k] = None


def reset_business_state():
    """
    清空全部业务数据（双端隔离）：
    切换角色（student <-> admin）时调用，保证两端不共享学习会话数据。
    只清业务键，保留身份（user_name/role）与导航（page/last_role）。
    """
    # 学习会话数据
    st.session_state.doc = {
        "filename": None, "chunks": [], "knowledge_candidates": [], "graph": None,
    }
    st.session_state.file_id = None
    st.session_state.index = None
    st.session_state.retriever = None
    st.session_state.chat_history = []
    st.session_state.last_qa = None
    # 智能问答多会话：角色切换时一并清空，防止跨端残留会话上下文
    st.session_state.qa_session_id = None
    st.session_state.qa_session_name = None
    st.session_state.quiz = None
    st.session_state.report = None
    st.session_state.path = None
    st.session_state.planner_result = None   # AI 自主规划结果随角色切换一并清空
    # 流水线 / 上传中间态（防止把未完成的解析任务带进另一端）
    st.session_state.uploaded_file = None
    st.session_state.uploaded_fp = None
    st.session_state.store_paths = {}   # 物理存储路径映射随课件一并清空
    st.session_state.upload_queue = []
    st.session_state.upload_total = 0
    st.session_state.batch_results = []
    st.session_state.parsing = False
    st.session_state.parse_stage = 0
    st.session_state.graph_ok = False
    st.session_state.graph_prev = None            # 上一份课件的图谱（增量提示数据源）
    st.session_state.graph_merge_pending = False  # 新文档待选择图谱更新方式
    st.session_state.parse_error = None
    st.session_state.parse_result_msg = None
    # 评估与作答输入
    reset_eval_state()
    clear_quiz_inputs("quiz")
    clear_quiz_inputs("pre")
    clear_quiz_inputs("post")


def clear_quiz_inputs(prefix="quiz", limit=50):
    """清理作答 radio 的残留选择（key 形如 quiz_0 / pre_1），恢复历史时避免显示旧答案"""
    for i in range(limit):
        st.session_state.pop(f"{prefix}_{i}", None)


# ========== 历史状态恢复：课件产物序列化 + 回调 ==========
def graph_to_dict(graph):
    """networkx.DiGraph -> 可 JSON 序列化的 dict（节点属性 + 边属性），供入库与恢复"""
    if graph is None:
        return None
    return {
        "nodes": [[str(n), dict(d)] for n, d in graph.nodes(data=True)],
        "edges": [[str(u), str(v), dict(d)] for u, v, d in graph.edges(data=True)],
    }


def graph_from_dict(data):
    """graph_to_dict 的逆操作：dict -> networkx.DiGraph（历史恢复用，失败返回 None）"""
    if not isinstance(data, dict):
        return None
    try:
        g = nx.DiGraph()
        for name, attrs in data.get("nodes") or []:
            g.add_node(str(name), **(attrs if isinstance(attrs, dict) else {}))
        for u, v, attrs in data.get("edges") or []:
            if g.has_node(str(u)) and g.has_node(str(v)):
                g.add_edge(str(u), str(v), **(attrs if isinstance(attrs, dict) else {}))
        return g
    except (TypeError, ValueError):
        return None


def doc_payload():
    """把当前课件的解析产物打包成可入库的 dict（files.data 列）"""
    doc = st.session_state.doc
    return {
        "chunks": doc["chunks"],
        "knowledge_candidates": doc["knowledge_candidates"],
        "graph": graph_to_dict(doc.get("graph")),
    }


def load_doc_from_db(file_id):
    """
    从数据库恢复课件上下文（历史状态恢复的核心步骤）：
      chunks/知识点候选/图谱 -> session_state，并本地重建检索索引（秒级、零 API）。
    安全：get_file_data 按当前用户 user_name+role 双重过滤，
    传入他人（或他角色）的 file_id 会归属不符 -> 按"文件丢失"处理，防数据串号。
    :return: True 恢复成功；False 课件记录不存在/归属不符/产物缺失
    """
    data = get_file_data(file_id,
                         user_name=st.session_state.user_name,
                         role=st.session_state.role)
    chunks = (data or {}).get("chunks") or []
    if not chunks:   # 无文本块 = 无法支撑问答/出题，按丢失处理
        return False
    doc = st.session_state.doc
    doc["filename"] = get_filename_of(file_id,
                                      user_name=st.session_state.user_name,
                                      role=st.session_state.role)
    doc["chunks"] = chunks
    doc["knowledge_candidates"] = (data or {}).get("knowledge_candidates") or []
    doc["graph"] = graph_from_dict((data or {}).get("graph"))
    # 本地重建 TF-IDF 检索索引（不依赖任何 API，保证恢复即时完成）
    index = IndexAgent().build([{"id": c["id"], "text": c["text"]} for c in chunks])
    st.session_state.index = index
    st.session_state.retriever = RetrieverAgent(index)
    return True


def get_filename_of(file_id, user_name=None, role=None):
    """按 file_id 查文件名（带用户归属过滤，查不到/非本人返回 None）"""
    for f in list_files(user_name=user_name, role=role):
        if f["id"] == file_id:
            return f["filename"]
    return None


# 历史记录类型 -> 恢复后跳转的页面
HISTORY_TARGET_PAGE = {
    "file": "上传解析",
    "qa": "智能问答",
    "diagnosis": "学习诊断",
    "path": "路径推荐",
    "eval": "学习评估",
}


@safe_callback
def restore_history(record):
    """
    侧边栏"最近学习记录"按钮回调：把该条记录关联的数据写回会话状态，
    并把导航切到对应页面，实现"历史状态恢复"。
    关联课件的解析产物无法还原时（原始文件已丢失），仅恢复记录本身内容，
    并给出警告提示，保证页面不崩溃。
    整个函数由 @safe_callback 兜底：任何意外异常都不会让页面出现红色 Traceback。
    """
    rtype = record["type"]
    d = record.get("detail") or {}
    file_id = record.get("file_id")

    # ---- 1) 尝试恢复课件上下文（chunks/知识点/图谱/检索索引） ----
    if file_id and load_doc_from_db(file_id):
        st.session_state.file_id = file_id
        st.session_state.uploaded_file = None      # 无需重新上传，清掉流水线缓存
        st.session_state.uploaded_fp = None
        st.session_state.upload_queue = []         # 清掉未完成的批量解析队列
        st.session_state.upload_total = 0
        st.session_state.batch_results = []
        st.session_state.history_warning = None
    else:
        # 原始文件已丢失：课件状态清空（文件名保留用于展示），只展示记录内容
        st.session_state.doc = {
            "filename": record.get("filename"), "chunks": [],
            "knowledge_candidates": [], "graph": None,
        }
        st.session_state.file_id = None
        st.session_state.index = None
        st.session_state.retriever = None
        st.session_state.history_warning = (
            f"原始文件已丢失，无法恢复「{record.get('filename') or '未知课件'}」的课件上下文，"
            "仅展示该条记录内容；如需继续学习请重新上传课件。"
        )

    # ---- 2) 按记录类型恢复业务数据 ----
    if rtype == "qa":
        # 恢复为一轮完整对话气泡（来源片段与知识点在旧记录中未存全文，留空展示）
        st.session_state.chat_history = [
            {"role": "user", "content": d.get("question", "")},
            {"role": "assistant", "content": d.get("answer", ""),
             "contexts": [], "knowledge_points": []},
        ]
        st.session_state.last_qa = None
        # 旧版单轮记录不属于任何多会话：重置会话态，之后提问自动开新会话
        st.session_state.qa_session_id = None
        st.session_state.qa_session_name = None
    elif rtype == "diagnosis":
        # 题目（含正确答案，供报告展示）与报告 JSON 逐一还原
        try:
            st.session_state.quiz = json.loads(d.get("questions") or "[]")
        except json.JSONDecodeError:
            st.session_state.quiz = []
        try:
            st.session_state.report = (json.loads(d["report"])
                                       if isinstance(d.get("report"), str) else d.get("report"))
        except json.JSONDecodeError:
            st.session_state.report = None
        st.session_state.path = None
        clear_quiz_inputs("quiz")
    elif rtype == "path":
        try:
            st.session_state.path = json.loads(d.get("path_text") or "[]")
        except json.JSONDecodeError:
            st.session_state.path = None
        # 路径页依赖诊断报告，回填同课件最近一份报告（没有也能只展示路径列表）
        st.session_state.report = _find_latest_report(file_id)
        st.session_state.quiz = None
        clear_quiz_inputs("quiz")
    elif rtype == "eval":
        st.session_state.evaluation = {
            "pre_score": d.get("pre_score", 0),
            "post_score": d.get("post_score", 0),
            "full_score": 100,   # eval 表未存满分，展示用默认值（报告文本含原始信息）
            "alg": d.get("alg", 0.0) or 0.0,
            "report": d.get("report") or "",
        }
        st.session_state.quiz = None
        st.session_state.report = None
        st.session_state.path = None
    # file 类型：课件已在步骤 1 恢复，无需额外数据

    # ---- 3) 重置与恢复无关的中间状态（解析流水线 / 评测中间卷） ----
    st.session_state.parsing = False
    st.session_state.parse_stage = 0
    st.session_state.parse_error = None
    st.session_state.parse_result_msg = None
    st.session_state.pre_quiz = None
    st.session_state.pre_score = None
    st.session_state.post_quiz = None
    st.session_state.post_score = None
    clear_quiz_inputs("pre")
    clear_quiz_inputs("post")

    # ---- 4) 自动切换导航到对应页面（radio 已绑定 key="page"） ----
    st.session_state.page = HISTORY_TARGET_PAGE.get(rtype, "历史记录")


def _find_latest_report(file_id):
    """
    取某课件最近一次诊断报告 dict（路径恢复时回填用；找不到返回 None）。
    安全：只查当前登录用户自己的历史（user_name+role 双重过滤）。
    """
    for r in get_history(st.session_state.user_name, st.session_state.role):
        if r["type"] == "diagnosis" and r.get("file_id") == file_id:
            try:
                rep = r["detail"].get("report")
                return json.loads(rep) if isinstance(rep, str) else rep
            except json.JSONDecodeError:
                continue
    return None


def _history_summary(record, max_len=14):
    """
    侧边栏"按课件归档"子项的摘要文本（只取关键字段，超长截断）：
      qa -> 首个问题；diagnosis -> 答对题数；path -> 阶段数；eval -> 前测→后测分。
    :param record: get_history / get_history_grouped 输出的统一格式记录
    :param max_len: 文本类摘要（问答）的最大字符数
    """
    rtype = record["type"]
    d = record.get("detail") or {}
    if rtype == "qa":
        # 压缩空白后截断，例如"这篇讲什么？"
        text = " ".join(str(d.get("question") or "查看问答").split())
    elif rtype == "diagnosis":
        score = d.get("score")
        return f"答对 {score} 题" if score is not None else "查看报告"
    elif rtype == "path":
        # path_text 是阶段列表 JSON，取阶段数做摘要（结构变化也不会崩）
        try:
            stages = json.loads(d.get("path_text") or "[]")
            return f"{len(stages)} 个阶段" if isinstance(stages, list) and stages else "查看路径"
        except json.JSONDecodeError:
            return "查看路径"
    elif rtype == "eval":
        return f"{d.get('pre_score', 0)}→{d.get('post_score', 0)} 分"
    else:
        text = "查看记录"
    return text[:max_len] + ("…" if len(text) > max_len else "")


# ========== AI 内容来源标注工具（所有 AI 生成内容必须绑定来源） ==========
def format_source_page(source_page=None, pages=None):
    """
    来源页码文本：优先用 AI 声明的页码，其次用片段自身的页码列表兜底。
    :param source_page: AI 返回的依据页码（int，可空）
    :param pages: 片段所在 PDF 页码列表（如 [3] 或 [3,4]，可空）
    :return: 「第3页」/「第3-4页」；无页码信息返回 None
    """
    if source_page:
        return f"第{source_page}页"
    if pages:
        ps = sorted({int(p) for p in pages if p})
        if ps:
            return f"第{ps[0]}页" if len(ps) == 1 else f"第{ps[0]}-{ps[-1]}页"
    return None


def build_source_line(source_page=None, source_snippet=None, pages=None):
    """
    把来源信息拼成 Markdown 引用行（AI 回答入库存库用）：
    返回 '' 或 '\\n\\n> 来源：第X页 - 片段'——历史记录回显时自带来源，无需数据迁移。
    """
    parts = []
    page_txt = format_source_page(source_page, pages)
    if page_txt:
        parts.append(page_txt)
    snip = str(source_snippet).strip() if source_snippet else ""
    if snip:
        parts.append(snip[:60])   # 片段截断 60 字，避免来源行喧宾夺主
    return f"\n\n> 来源：{' - '.join(parts)}" if parts else ""


def render_quiz_answers(quiz, key_prefix):
    """
    渲染选择题作答区（诊断页/评估页共用）。
    防泄漏：渲染数据只保留题干/考点/选项，剔除 answer（正确答案）与
    explain（解析）字段——用户未提交前，严禁向前端展示任何对错线索。
    """
    for i, raw in enumerate(quiz):
        q = {k: v for k, v in raw.items() if k not in ("answer", "explain")}
        st.markdown(f"**第 {i + 1} 题：{q['question']}**")
        st.caption(f"考点：{q.get('knowledge_point', '未知知识点')}")
        # AI 出题来源小字（依据的课件页码/原文片段）：不含对错信息，无泄漏风险
        page_txt = format_source_page(raw.get("source_page"))
        snip = str(raw.get("source_snippet") or "").strip()
        if page_txt or snip:
            parts = [p for p in (page_txt, f"「{snip[:60]}」" if snip else "") if p]
            st.caption("来源：" + " - ".join(parts))
        labels = [f"{k}. {v}" for k, v in q["options"].items()]
        st.radio("选项", options=labels, key=f"{key_prefix}_{i}", label_visibility="collapsed")


def collect_answers(quiz, key_prefix):
    """收集作答结果（界面选项形如 'A. xxx'，取首字母还原为选项字母）"""
    return [st.session_state.get(f"{key_prefix}_{i}", "")[:1] for i in range(len(quiz))]


def clear_doc_state():
    """用户移除课件后，清空所有与之相关的会话状态（含解析流水线状态）"""
    st.session_state.doc = {"filename": None, "chunks": [], "knowledge_candidates": [], "graph": None}
    st.session_state.file_id = None
    st.session_state.index = None
    st.session_state.retriever = None
    st.session_state.uploaded_file = None
    st.session_state.uploaded_fp = None
    st.session_state.store_paths = {}   # 物理存储路径映射随课件一并清空
    st.session_state.upload_queue = []
    st.session_state.upload_total = 0
    st.session_state.batch_results = []
    st.session_state.parsing = False
    st.session_state.parse_stage = 0
    st.session_state.parse_error = None
    st.session_state.parse_result_msg = None
    st.session_state.graph_prev = None            # 课件已移除：增量提示一并失效
    st.session_state.graph_merge_pending = False
    st.session_state.fp = None
    st.session_state.quiz = None
    st.session_state.report = None
    st.session_state.path = None
    st.session_state.last_qa = None
    # 课件已移除：问答会话按课件隔离，一并重置（下次提问自动开新会话）
    st.session_state.qa_session_id = None
    st.session_state.qa_session_name = None
    reset_eval_state()


def eval_to_markdown(ev):
    """把学习增益评估结果 dict 转成 Markdown 文本（用于存库与展示）"""
    return (
        f"**前测：{ev['pre_score']}/{ev['full_score']} → 后测：{ev['post_score']}/{ev['full_score']}**\n\n"
        f"**学习增益 ALG：{ev['alg']:.2f}**\n\n"
        f"{ev['report']}"
    )


def build_graph_with_spinner():
    """调用 OntologyAgent 构建知识图谱（返回 networkx.DiGraph）；成功则更新状态并返回 True。
    失败时在此处统一显示用户端安全提示（AI 服务 / 网络问题），调用方无需重复报错。"""
    # 开发者端开关：图谱 Agent 被停用时拒绝构建
    # （解析流水线第 4 步 / 图谱页重建 / 诊断与评估的补图，全部经由本函数，一处拦截全局生效）
    if not get_agent_config()["ontology_enabled"]:
        st.warning("管理员已停用「图谱 Agent」，无法构建知识图谱。")
        return False
    # 轻量限流（需求 2）：图谱构建也是 AI 请求，纳入每分钟 5 次配额
    if not ai_rate_limit_check():
        return False
    doc = st.session_state.doc
    agent = OntologyAgent()
    # 限时 90s + 失败降级：图谱构建失败不崩溃，文本块与问答功能不受影响
    graph = call_agent_limited(
        lambda: agent.build(doc["knowledge_candidates"], chunks=doc["chunks"]),
        agent=agent, timeout=90, spinner_text="AI 正在构建知识图谱...")
    if graph is not None and graph.number_of_nodes() > 0:
        st.session_state.doc["graph"] = graph
        return True
    # 局部降级（需求 2）：图谱只是课件的增强层，失败不阻断学习主流程
    st.warning("知识图谱构建失败，但已提取文本块——「智能问答」等功能不受影响，稍后可在本页重试构建。")
    return False


# ========== 页面 1：上传与解析（分阶段流水线，支持断点续传） ==========
MAX_UPLOAD_MB = 20   # 单文件上传大小上限（MB）：与 .streamlit/config.toml 的 maxUploadSize 保持一致


def _reset_downstream_state():
    """清空依赖旧课件的所有下游结果（题目/报告/路径/问答/评估）"""
    st.session_state.quiz = None
    st.session_state.report = None
    st.session_state.path = None
    st.session_state.last_qa = None
    reset_eval_state()


def _mark_task(status, error_msg=None):
    """标记当前解析任务状态（tasks 表）；无任务 id 时静默跳过（旧会话升级兼容）"""
    tid = st.session_state.get("task_id")
    if tid:
        update_task_status(tid, status, error_msg)


def _start_parse():
    """
    启动解析流水线（两阶段启动的关键）：
    本函数只设置状态并触发 rerun —— 下一次脚本运行会先以「禁用」状态渲染侧边栏，
    然后才进入耗时的解析步骤，从而保证用户在解析期间无法切换页面。
    """
    # 开发者端开关：解析 Agent 被停用时拒绝启动流水线（提示经 parse_result_msg 显示一次）
    if not get_agent_config()["parser_enabled"]:
        st.session_state.parse_result_msg = (
            "warning", "管理员已停用「解析 Agent」，无法开始解析。")
        return
    f = st.session_state.uploaded_file
    # 增量图谱：暂存上一份课件的图谱（仅单文件上传时提示增量更新；批量模式逐文件独立构建）
    prev_g = st.session_state.doc.get("graph")
    st.session_state.graph_prev = prev_g if (
        prev_g is not None and prev_g.number_of_nodes() > 0) else None
    st.session_state.graph_merge_pending = False
    st.session_state.kp_extract_failed = False   # 新一轮解析：复位知识点抽取降级标志
    # 任务追踪（需求 2）：登记一条 pending 任务，流水线开始执行时转 processing
    st.session_state.task_id = create_task("parse_pdf",
                                           st.session_state.user_name, st.session_state.role)
    st.session_state.doc = {
        "filename": f.name, "chunks": [], "knowledge_candidates": [], "graph": None,
    }
    st.session_state.index = None
    st.session_state.retriever = None
    _reset_downstream_state()
    st.session_state.batch_results = []   # 批次结果从零开始累计
    st.session_state.parse_stage = 1   # 从步骤 1 开始
    st.session_state.parsing = True    # 禁用侧边栏导航
    st.rerun()


def _abort_parse(msg):
    """解析流水线异常终止：记录错误信息、解除侧边栏禁用（重新解析需重新点击按钮）"""
    _mark_task("failed", msg)   # 任务追踪：整条流水线失败（含失败原因）
    st.session_state.parse_error = msg
    st.session_state.parsing = False
    st.session_state.parse_stage = 0
    st.rerun()


def _load_next_batch_file():
    """
    批量解析队列衔接：还有待解析文件时装载下一个并回到步骤 1（返回 True），
    队列已空（批次结束）返回 False。装载即"换文件"：重置课件产物与下游结果，
    保证每个文件的解析互不污染。
    """
    queue = st.session_state.upload_queue
    if not queue:
        return False
    f = queue.pop(0)
    st.session_state.uploaded_file = f
    st.session_state.graph_prev = None   # 批量模式不做跨文件增量提示，每个文件独立构建
    st.session_state.kp_extract_failed = False   # 每个文件的降级标志独立，装载新文件时复位
    # 任务追踪：批量模式下每个文件一条独立任务（逐文件登记 pending）
    st.session_state.task_id = create_task("parse_pdf",
                                           st.session_state.user_name, st.session_state.role)
    st.session_state.doc = {
        "filename": f.name, "chunks": [], "knowledge_candidates": [], "graph": None,
    }
    st.session_state.index = None
    st.session_state.retriever = None
    _reset_downstream_state()
    st.session_state.parse_stage = 1
    return True


def _finalize_batch():
    """批次全部结束（全部成功 / 个别失败）：解除侧边栏禁用并生成汇总提示"""
    st.session_state.parse_stage = 0
    st.session_state.parsing = False
    results = st.session_state.batch_results
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    if st.session_state.upload_total <= 1:
        # 单文件模式：沿用原有的成功/警告提示
        r = ok[0]
        if st.session_state.get("kp_extract_failed"):
            # 局部降级（需求 2）：知识点抽取失败 -> 明确告知"文本块可用、问答不受影响"
            st.session_state.parse_result_msg = (
                "warning", f"解析完成：{r['info']}；AI 知识点抽取未成功，已跳过图谱构建——"
                           "「智能问答」不受影响，可重新解析本文件补充知识点。")
        elif st.session_state.graph_merge_pending:
            # 已有图谱 + 新文档：不自动构建，等用户在上传页选择增量/全新
            st.session_state.parse_result_msg = (
                "success", f"解析完成：{r['info']}；检测到已有知识图谱，可在下方选择增量更新。")
        elif st.session_state.graph_ok:
            if get_agent_config()["ontology_enabled"]:
                st.session_state.parse_result_msg = (
                    "success", f"解析完成：{r['info']}，知识图谱已自动构建！")
            else:
                # 图谱 Agent 停用：info 已含"跳过图谱构建"说明，成功提示不再追加图谱字样
                st.session_state.parse_result_msg = ("success", f"解析完成：{r['info']}。")
        else:
            st.session_state.parse_result_msg = (
                "warning", f"解析完成：{r['info']}；知识图谱构建失败，可在下方重试。")
    else:
        # 批量模式：逐文件列出结果，成功在前、失败在后
        lines = [f"**{r['filename']}**（{r['info']}）" for r in ok]
        lines += [f"**{r['filename']}**：{r['info']}" for r in fail]
        kind = "success" if not fail else "warning"
        st.session_state.parse_result_msg = (
            kind, f"批量解析完成：成功 {len(ok)} / 失败 {len(fail)}（共 {len(results)} 个）。\n\n"
                  + "\n\n".join(lines))
    st.rerun()


def _skip_failed_file(err_msg):
    """
    批量模式下单文件解析失败：记录失败结果后自动衔接下一个文件；
    如果是最后一个文件则直接收尾。单文件模式交由调用方走 _abort_parse。
    """
    _mark_task("failed", err_msg)   # 任务追踪：本文件解析失败（含失败原因）
    st.session_state.batch_results.append(
        {"filename": st.session_state.doc["filename"], "ok": False, "info": err_msg})
    if _load_next_batch_file():
        st.rerun()
    _finalize_batch()


def run_parse_pipeline():
    """
    解析流水线：按 parse_stage 从断点继续执行（内嵌 st.status 实时进度面板）。
    每个阶段完成后立即把结果写入 session_state 并 rerun —— 即使脚本在阶段中途
    被意外中断，已完成的阶段结果也已持久化，重跑时自动从断点继续，无需重新上传。
    面板内容：总进度条 + 步骤清单（已完成/进行中/待执行）+ 当前阶段实时进度。
    """
    doc = st.session_state.doc
    f = st.session_state.uploaded_file
    stage = st.session_state.parse_stage
    # 批次进度：当前是第几个文件（total - 剩余队列长度）；单文件时恒为 1/1
    total = max(st.session_state.upload_total, 1)
    idx = total - len(st.session_state.upload_queue)
    file_tag = f"第 {idx}/{total} 个「{doc['filename']}」" if total > 1 else f"「{doc['filename']}」"
    # 四个解析步骤（与 parse_stage 1-4 对应；stage 5 为入库收尾，瞬间完成不单独展示）
    step_names = ["提取文本并切块", "AI 抽取知识点", "构建检索索引", "构建知识图谱"]
    try:
        with st.status(f"正在解析 {file_tag}", state="running", expanded=True):
            # ---- 总进度条：让用户对整体耗时心中有数 ----
            overall = (min(stage, 5) - 1) / 4
            st.progress(overall, text=f"总进度 {int(overall * 100)}%")
            # ---- 步骤清单：已完成打勾、当前高亮、未执行置灰 ----
            for i, name in enumerate(step_names, start=1):
                if stage == 5 or stage > i:
                    st.markdown(f"{name}")
                elif stage == i:
                    st.markdown(f"**{name}**")
                else:
                    st.markdown(f"<small>{name}</small>", unsafe_allow_html=True)
            # ---- 当前阶段的实时进度控件（由下方回调就地更新，无需 rerun） ----
            page_bar = batch_bar = None
            if stage == 1:
                page_bar = st.progress(0.0, text="正在提取文本… 0%")
            elif stage == 2:
                batch_bar = st.progress(0.0, text="正在抽取知识点… 0%")
            elif stage == 3:
                st.progress(0.6, text="正在构建检索索引（TF-IDF 向量化）…")
            elif stage == 4:
                st.progress(0.8, text="正在构建知识图谱（AI 生成知识点依赖）…")
            # ---- 排队提示：批量模式下列出尚未解析的文件 ----
            if st.session_state.upload_queue:
                q = st.session_state.upload_queue
                preview = "、".join(x.name for x in q[:4]) + ("…" if len(q) > 4 else "")
                st.caption(f"排队中：还有 {len(q)} 个文件待解析：{preview}")

            if stage == 1:   # 步骤 1/4：按扩展名提取文本并切块（纯本地，PDF 逐页回调）
                _mark_task("processing")   # 任务追踪：流水线开始执行（pending -> processing）
                def _page_cb(done, n_pages):
                    pct = int(done / n_pages * 100) if n_pages else 100
                    page_bar.progress(done / n_pages if n_pages else 1.0,
                                      text=f"正在解析第 {done}/{n_pages} 页… {pct}%")
                f.seek(0)   # 文件指针复位（重跑恢复的文件对象指针可能不在开头）
                doc["chunks"] = ParserAgent().extract_chunks(f, progress_cb=_page_cb)
                st.session_state.parse_stage = 2
                st.rerun()
            if stage == 2:   # 步骤 2/4：AI 抽取知识点（最耗时，逐批回调实时刷新已抽取数量）
                def _batch_cb(done_b, n_b, found):
                    pct = int(done_b / n_b * 100) if n_b else 100
                    batch_bar.progress(done_b / n_b if n_b else 1.0,
                                       text=f"正在抽取知识点：{done_b}/{n_b} 批（{pct}%），已累计 {found} 个")
                doc["knowledge_candidates"] = ParserAgent().extract_knowledge(
                    doc["chunks"], progress_cb=_batch_cb)
                # 局部降级（需求 2）：AI 知识点抽取全失败（网络/限流等）时不终止流水线——
                # 文本块已提取，问答检索照常可用；后续步骤 4 跳过图谱构建并给出降级提示
                if not doc["knowledge_candidates"]:
                    print("[ERROR parse] knowledge extraction returned 0 candidates -> degrade")
                    st.session_state.kp_extract_failed = True
                st.session_state.parse_stage = 3
                st.rerun()
            if stage == 3:   # 步骤 3/4：本地检索索引（秒级）
                st.session_state.index = IndexAgent().build(doc["chunks"])
                st.session_state.retriever = RetrieverAgent(st.session_state.index)
                st.session_state.parse_stage = 4
                st.rerun()
            if stage == 4:   # 步骤 4/4：构建知识图谱（已有图谱时不自动全量重建，交由用户选择省 API）
                prev_g = st.session_state.get("graph_prev")
                if st.session_state.get("kp_extract_failed"):
                    # 知识点未抽出：无图谱可建，按降级处理（graph_ok=False 走收尾降级文案）
                    st.session_state.graph_ok = False
                elif prev_g is not None:
                    # 检测到已有图谱：解析完成后在上传页提示"增量更新 / 全新构建 / 跳过"，
                    # 增量模式只让 AI 处理新知识点（旧节点与旧边原样保留），避免全量重建的 API 开销
                    st.session_state.graph_merge_pending = True
                    st.session_state.graph_ok = True   # 收尾消息不走"构建失败"分支
                elif get_agent_config()["ontology_enabled"]:
                    st.session_state.graph_ok = build_graph_with_spinner()
                else:
                    # 图谱 Agent 已被管理员停用：解析照常完成，仅跳过图谱构建
                    # （graph_ok=True 走成功分支；收尾 info 会注明"跳过图谱构建"，避免误报）
                    st.session_state.graph_ok = True
                st.session_state.parse_stage = 5
                st.rerun()
        if stage == 5:   # 收尾：课件入库（含解析产物与物理存储路径）、更新指纹；批量模式下自动衔接下一个文件
            st.session_state.file_id = save_file(
                doc["filename"], data=doc_payload(),
                user_name=st.session_state.user_name, role=st.session_state.role,
                store_path=(st.session_state.get("store_paths") or {}).get(doc["filename"]),
            )
            log_event("upload", f"解析完成「{doc['filename']}」（{len(doc['chunks'])} 个文本块），已加入用户专属知识库",
                      user_name=st.session_state.user_name, role=st.session_state.role)
            # 记录本文件结果：批量结束时用于生成汇总提示
            # （图谱 Agent 停用时在 info 里注明，避免收尾提示误报"已构建图谱"）
            _info = f"{len(doc['chunks'])} 个文本块、{len(doc['knowledge_candidates'])} 个知识点候选"
            if st.session_state.get("kp_extract_failed"):
                _info += "；知识点抽取失败，已跳过图谱（问答不受影响）"
            elif not get_agent_config()["ontology_enabled"] and doc.get("graph") is None:
                _info += "；图谱 Agent 已停用，跳过图谱构建"
            st.session_state.batch_results.append({
                "filename": doc["filename"], "ok": True, "info": _info,
            })
            _mark_task("success")   # 任务追踪：本文件解析成功
            # 队列里还有文件：装载下一个并回到步骤 1（分批处理，每轮脚本只推进一个阶段，页面不卡死）
            if _load_next_batch_file():
                st.rerun()
            # 批次结束：生成完成提示、解除侧边栏禁用
            st.session_state.fp = st.session_state.uploaded_fp
            _finalize_batch()
    except (EncryptedPDFError, ScannedPDFError, ParseError) as e:
        # 业务语义异常：异常消息本身是面向用户的友好中文（见 parser_agent.py），可直接展示。
        # 批量模式：跳过失败文件继续下一个；单文件模式：整条流水线终止。
        if st.session_state.upload_total > 1:
            _skip_failed_file(str(e))
        _abort_parse(str(e))
    except Exception:   # 兜底（API/网络等未知异常）：用户端只给安全文案，堆栈进开发者终端
        print("[ERROR parse] Parse pipeline crashed unexpectedly")
        error_logger.error("解析流水线未捕获异常（stage=%s, file=%s）",
                           st.session_state.parse_stage, doc.get("filename"), exc_info=True)
        _abort_parse("解析过程中出现问题，请更换文件或稍后重试。")


def page_upload():
    # ---- 学习端首页：个性化欢迎语 + 当前用户统计卡片 ----
    # 防 XSS：昵称/文件名等用户输入进入 unsafe HTML 渲染前必须 html.escape
    st.markdown(
        f'<p style="font-size:1.05rem; color:#86868b; margin-bottom:0;">'
        f'欢迎回来，<b>{html.escape(str(st.session_state.user_name))}</b></p>',
        unsafe_allow_html=True,
    )
    st.title("上传与解析")
    st.caption(
        "上传 PDF / Word / TXT / Markdown 课件（支持多选批量解析）：自动提取文本块、抽取知识点、"
        "构建检索索引并生成知识图谱。\n\n"
        "解析分 4 步进行，每个文件完成即保存进度并自动衔接下一个；解析期间侧边栏暂时禁用，请勿刷新页面。"
    )

    # ---- 学习统计卡片（数据来自 SQLite 历史记录，打开首页即见） ----
    history = get_history(st.session_state.user_name, st.session_state.role)   # 只统计当前用户的学习数据
    n_files = len({r["filename"] for r in history})               # 上传过的课件数（按文件名去重）
    n_qa = sum(1 for r in history if r["type"] == "qa")           # 累计问答次数
    n_diag = sum(1 for r in history if r["type"] == "diagnosis")  # 累计诊断次数
    c1, c2, c3 = st.columns(3)
    c1.metric("已上传文件", n_files)
    c2.metric("累计问答", n_qa)
    c3.metric("累计诊断", n_diag)
    st.divider()

    # ---- 文件管理：历史文件列表（一键加载 / 删除并级联清理关联记录） ----
    st.subheader("文件管理")
    files = list_files(user_name=st.session_state.user_name, role=st.session_state.role)
    if not files:
        st.caption("暂无历史文件。解析完成后课件会自动保存到这里，可随时一键恢复。")
    for f in files:
        n_records = f["qa"] + f["diagnosis"] + f["path"] + f["eval"]
        col_a, col_b, col_c = st.columns([5, 1, 1])
        current = " · **当前**" if f["id"] == st.session_state.get("file_id") else ""
        col_a.markdown(
            f"**{html.escape(str(f['filename']))}**{current}  \n"
            f"<small>上传：{f['upload_time']} · 用户：{html.escape(str(f['user_name']))}"
            f" · 关联记录：{n_records} 条</small>",
            unsafe_allow_html=True,
        )
        if col_b.button("加载", key=f"load_{f['id']}", use_container_width=True,
                        disabled=st.session_state.parsing):
            if load_doc_from_db(f["id"]):
                st.session_state.file_id = f["id"]
                st.session_state.uploaded_file = None
                st.session_state.uploaded_fp = None
                st.session_state.parse_stage = 0
                st.session_state.parsing = False
                st.toast(f"已加载「{f['filename']}」，可前往问答 / 诊断 / 图谱页继续学习。")
            else:
                st.error("该文件的解析产物已丢失，无法加载，请重新上传。")
        if col_c.button("删除", key=f"del_{f['id']}", use_container_width=True):
            # 弹出删除确认框（st.dialog 模态窗）：目标文件信息先入 session_state 再打开，
            # 与侧边栏"删除会话"弹窗的传参方式一致，防止误删
            st.session_state.dialog_file = {
                "id": f["id"], "filename": f["filename"],
                "n_records": n_records, "store_path": f.get("store_path"),
            }
            delete_file_dialog()
    st.divider()

    # ---- 流水线消息（完成提示 / 错误提示，均用 st.status 呈现，显示一次即清除） ----
    if st.session_state.parse_result_msg:
        kind, text = st.session_state.parse_result_msg
        label = "解析完成" if kind == "success" else "解析完成（部分文件失败）"
        with st.status(label, state="complete", expanded=True):
            st.markdown(text)
        # 需求 1：解析完成 Toast（成功绿色 / 部分失败红色）
        st.toast(label)   # 解析完成 Toast（成功 / 部分失败共用同一文案）
        st.session_state.parse_result_msg = None
    if st.session_state.parse_error:
        with st.status("解析失败", state="error", expanded=True):
            st.markdown(st.session_state.parse_error)
        st.session_state.parse_error = None

    # ---- 增量更新知识图谱提示：新文档解析完成且检测到已有图谱时出现（避免全量重建省 API） ----
    if st.session_state.get("graph_merge_pending"):
        prev_g = st.session_state.get("graph_prev")
        _doc = st.session_state.doc
        st.info(
            f"检测到已有知识图谱（{prev_g.number_of_nodes()} 个知识点 / "
            f"{prev_g.number_of_edges()} 条关系），新文档「{_doc['filename']}」已解析完成"
            f"（{len(_doc['knowledge_candidates'])} 个知识点候选）。如何更新知识图谱？"
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("增量更新（推荐）", type="primary", use_container_width=True,
                     help="保留已有知识点与关系，只让 AI 分析新增的知识点，节省 API 成本"):
            if not get_agent_config()["ontology_enabled"]:
                agent_disabled_msg("「图谱 Agent」")   # 停用时保留待选状态，用户可改选"跳过"或等管理员开启
            elif not ai_rate_limit_check():   # 轻量限流（需求 2）：增量更新计入配额
                return
            else:
                agent = OntologyAgent()
                # 限时 90s + 降级：超时/失败返回 None 时不解包，保留待选状态供重选
                _r = call_agent_limited(
                    lambda: agent.merge_build(
                        _doc["knowledge_candidates"], chunks=_doc["chunks"], existing_graph=prev_g),
                    agent=agent, timeout=90, spinner_text="AI 正在增量更新知识图谱（只处理新知识点）…")
                if _r is None:
                    st.warning("图谱增量更新失败（AI 服务繁忙或超时），"
                               "可点击重试，或选择「暂不更新」先去问答。")
                    return
                merged, added = _r
                st.session_state.graph_merge_pending = False
                st.session_state.graph_prev = None
                if merged is not None and merged.number_of_nodes() > 0:
                    st.session_state.doc["graph"] = merged
                    st.session_state.graph_ok = True
                    if st.session_state.get("file_id"):
                        update_file_data(st.session_state.file_id, doc_payload())   # 合并图回填入库（否则刷新后图谱丢失）
                    st.session_state.parse_result_msg = (
                        "success", f"知识图谱增量更新完成：新增 {added} 个知识点，"
                                   f"现有 {merged.number_of_nodes()} 个节点 / {merged.number_of_edges()} 条关系。")
                else:
                    st.session_state.parse_result_msg = (
                        "warning", "知识图谱增量更新失败，可在图谱页重试或选择全新构建。")
                st.rerun()
        if b2.button("为本文档全新构建", use_container_width=True,
                     help="忽略已有图谱，只为当前文档重新构建（原始行为）"):
            st.session_state.graph_merge_pending = False
            st.session_state.graph_prev = None
            if build_graph_with_spinner():
                if st.session_state.get("file_id"):
                    update_file_data(st.session_state.file_id, doc_payload())   # 全新构建的图同样回填入库
                st.session_state.parse_result_msg = ("success", "已为本文档全新构建知识图谱。")
                st.rerun()
        if b3.button("暂不更新", use_container_width=True,
                     help="暂不构建图谱（可在知识图谱页随时构建）"):
            st.session_state.graph_merge_pending = False
            st.session_state.graph_prev = None
            st.rerun()

    # ---- 文件上传组件（多选批量；文件对象与指纹持久化到 session_state，重跑/切页不丢） ----
    # —— 上传中断检测（需求 4）：.part 残留 = 上次传输/保存被中断的证据，友好提示 ——
    part_leftovers = storage.list_part_files(st.session_state.user_name)
    if part_leftovers:
        st.warning(
            "检测到上次上传被中断的残留文件：" + "、".join(f"`{p}`" for p in part_leftovers)
            + "。如需使用请重新上传该文件；残留文件可在管理员「系统维护」页一键清理。"
        )
    # ---- 版权合规提示（需求：隐私与版权合规——紧贴上传组件，上传前必见） ----
    st.caption("**请注意：请确保您拥有该文档的合法版权或使用权，切勿上传涉密、侵权或违规内容。**")
    uploaded_list = st.file_uploader(
        "选择课件文件（支持 PDF / Word / TXT / Markdown，可按住 Ctrl 多选批量上传）",
        type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
    ) or []
    if uploaded_list:
        # —— 单文件大小限制（20MB）：超限文件友好拦截，不落盘、不进入解析队列 ——
        # （Streamlit 层面另有 .streamlit/config.toml 的 maxUploadSize=20 双保险）
        oversized = [f for f in uploaded_list if f.size > MAX_UPLOAD_MB * 1024 * 1024]
        ok_list = [f for f in uploaded_list if f.size <= MAX_UPLOAD_MB * 1024 * 1024]
        if oversized:
            _names = "、".join(f"「{f.name}」（{f.size / 1048576:.1f} MB）" for f in oversized)
            st.error(
                f"文件过大，请上传小于{MAX_UPLOAD_MB}MB的文档。"
                f"以下文件已自动忽略：{_names}。"
                "仅支持 PDF / Word / TXT / Markdown，请压缩后重新上传。"
            )
        if not ok_list:
            # 全部超限：视为"没有有效上传"。记住该指纹避免每次重跑重复触发新批次判断
            if st.session_state.uploaded_fp != "":
                st.session_state.uploaded_fp = ""
                if not st.session_state.parsing and st.session_state.fp is not None:
                    clear_doc_state()   # 用户移除了原课件：清空下游状态
            return
        # 指纹 = 有效文件的名称+大小拼接（任何文件增删都会变化，触发重新缓存）
        fp = "|".join(f"{f.name}_{f.size}" for f in ok_list)
        if st.session_state.uploaded_fp != fp:
            # —— 物理落盘（多租户隔离）：每个上传文件立即存入当前用户专属目录 ——
            # 路径形如 data/{user_name}/{文件名}；只有当前用户能读写自己的目录。
            # 落盘失败的文件（昵称/文件名含非法字符等）跳过并提示，不阻断解析流程。
            store_paths = dict(st.session_state.get("store_paths") or {})
            # —— 落盘进度条（需求 4 断点续传体验）：分块写入 + 逐文件实时刷新进度 ——
            save_bar = st.progress(0.0, text="正在保存上传文件到专属存储…")
            for fi, _f in enumerate(ok_list):
                base = fi / len(ok_list)          # 本文件在整个批次中的起点
                span = 1 / len(ok_list)           # 本文件占据的进度条区间
                try:
                    store_paths[_f.name] = storage.save_original(
                        st.session_state.user_name, _f.name, _f.getvalue(),
                        progress_cb=lambda w, t, _b=base, _s=span, _n=_f.name:
                            save_bar.progress(min(_b + (w / t) * _s, 1.0),
                                              text=f"正在保存 {_n}（批次进度 {_b * 100 + (w / t) * _s * 100:.0f}%）…"))
                except (ValueError, OSError) as e:
                    # 中断/落盘失败友好提示：半截 .part 残留已自动回收，不影响其他文件
                    st.warning(f"文件「{_f.name}」保存中断或失败（{e}），可重新上传该文件。")
            save_bar.progress(1.0, text="上传文件已全部保存到专属存储")
            st.toast(f"已成功上传 {len(ok_list)} 个文件")   # 需求 1：上传成功 Toast
            st.session_state.store_paths = store_paths
            # 新一批文件：第一个立即进入流水线，其余进入待解析队列（分批逐个处理）
            st.session_state.uploaded_file = ok_list[0]
            st.session_state.upload_queue = list(ok_list[1:])
            st.session_state.upload_total = len(ok_list)
            st.session_state.batch_results = []
            st.session_state.uploaded_fp = fp
    else:
        # 解析中用户误点"清除"：忽略，沿用缓存继续；空闲状态移除文件则清空全部状态
        if not st.session_state.parsing:
            if st.session_state.fp is not None:
                clear_doc_state()
            return
        fp = st.session_state.uploaded_fp

    fp = st.session_state.uploaded_fp
    doc = st.session_state.doc

    # ---- 解析流水线运行中：从断点继续执行（st.status 进度面板内嵌于流水线函数） ----
    if st.session_state.parse_stage > 0:
        run_parse_pipeline()
        return

    # ---- 新文件待解析：排队状态面板 + 启动按钮（两阶段启动，保证侧边栏先禁用再生效） ----
    if fp is not None and fp != st.session_state.fp:
        names = [st.session_state.uploaded_file.name] + [x.name for x in st.session_state.upload_queue]
        with st.status("排队待解析", state="running", expanded=True):
            for i, n in enumerate(names, start=1):
                st.markdown(f"{i}. `{n}`")
            st.caption(
                f"共 {len(names)} 个文件。点击下方按钮开始解析：每个文件约需 1-2 分钟（含 AI 调用），"
                "将按顺序逐个处理，期间侧边栏暂时禁用。"
            )
        if st.button("开始解析", type="primary"):
            _start_parse()
        return

    if not doc["chunks"]:
        st.info("当前没有已解析的课件，请先上传 PDF / Word / TXT / Markdown 文件。")
        return

    # ---- 解析结果概览 ----
    m1, m2, m3 = st.columns(3)
    m1.metric("文本块", len(doc["chunks"]))
    m2.metric("知识点候选", len(doc["knowledge_candidates"]))
    m3.metric("知识图谱", "已生成" if doc["graph"] else "未生成")

    st.subheader("知识点候选")
    if doc["knowledge_candidates"]:
        st.markdown("、".join(f"`{k}`" for k in doc["knowledge_candidates"]))
    else:
        st.caption("暂无知识点候选")

    st.subheader("文本块预览")
    for c in doc["chunks"][:5]:
        with st.expander(f"{c['id']}（{len(c['text'])} 字）"):
            st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
    if len(doc["chunks"]) > 5:
        st.caption(f"其余 {len(doc['chunks']) - 5} 个文本块已建立索引，可在问答页检索。")

    # ---- 图谱重试入口（上传时图谱构建失败的情况；失败提示由 build_graph_with_spinner 统一显示） ----
    if doc["graph"] is None and doc["knowledge_candidates"]:
        if st.button("生成知识图谱", type="primary"):
            if build_graph_with_spinner():
                st.rerun()


# ========== 页面 2：智能问答 ==========
def build_qa_markdown(qa, filename=None):
    """
    把当前问答记录整理成 Markdown 学习笔记文本（供"导出笔记"按钮下载）。
    :param qa: last_qa 记录 {"question", "answer", "contexts", "knowledge_points"}
    :param filename: 当前课件名（作为笔记标题的一部分）
    """
    lines = [
        f"# 学习笔记：{filename or '智学 AI 学习助手'}",
        "",
        f"> 导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 问题",
        "",
        qa["question"],
        "",
        "## AI 回答",
        "",
        qa["answer"],
        "",
    ]
    # AI 回答的来源标注（依据的课件页码与原文片段）
    src_line = build_source_line(
        qa.get("source_page"), qa.get("source_snippet"),
        (qa.get("contexts") or [{}])[0].get("pages"),
    )
    if src_line:
        lines += [src_line.strip(), ""]
    # 相关知识点（混合检索的图谱路结果）
    kps = qa.get("knowledge_points") or []
    if kps:
        lines += ["## 相关知识点", ""]
        lines += [
            f"- **{p['name']}**（难度：{p['difficulty']}，{p['match_type']}）：{p['description'] or '暂无描述'}"
            for p in kps
        ]
        lines += [""]
    # 引用的课件片段（含相关度，正文截断 300 字防笔记过长）
    ctxs = qa.get("contexts") or []
    if ctxs:
        lines += ["## 来源片段", ""]
        lines += [
            f"> 【{c['id']}】（相关度 {c['score']:.2f}）{c['text'][:300]}{'...' if len(c['text']) > 300 else ''}"
            for c in ctxs
        ]
        lines += [""]
    return "\n".join(lines)


# ========== 智能问答多会话管理 ==========
def _new_qa_session():
    """新建问答会话：只初始化内存态（首条问答入库时才落库，避免产生空会话记录）"""
    st.session_state.qa_session_id = uuid.uuid4().hex[:12]
    st.session_state.qa_session_name = "新对话"
    st.session_state.chat_history = []
    st.session_state.last_qa = None
    _sync_sid_param()   # 需求 3：会话切换时地址栏 sid 参数同步更新


def _click_new_session():
    """
    新建对话按钮的 on_click 回调：初始化新会话 + 自动跳转问答页，
    让"新建"操作立刻可见（气泡清空 + 侧边栏出现"新对话"占位）。
    注意两点：
      - 回调先于 widget 实例化执行，此处写 st.session_state.page 安全
        （在脚本主体里写会报 "cannot be modified" 异常）；
      - on_click 回调执行完 Streamlit 会自动整页重跑，无需再调 st.rerun()。
    首条提问的懒落库路径（_handle_chat_question）复用 _new_qa_session，不带跳转。
    """
    _new_qa_session()
    st.session_state.page = "智能问答"


def _open_qa_session(session_id):
    """
    点击历史会话：从数据库恢复当时的完整对话气泡。
    恢复内容含来源字段、引用片段与命中知识点（存于 meta_json）；
    旧版记录无 meta -> 降级为纯文本气泡（仅问答正文）。
    """
    rows = get_session_qas(session_id, st.session_state.user_name, st.session_state.role)
    history = []
    for r in rows:
        history.append({"role": "user", "content": r["question"] or ""})
        # 解析该轮回答的完整上下文（老记录 meta_json 为空 -> 全部留空）
        try:
            meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        except json.JSONDecodeError:
            meta = {}
        # 入库时 answer 拼了来源引用行（供学习记录页回显），恢复气泡时剥掉，
        # 来源小字改由结构化字段渲染，避免重复显示
        answer = r["answer"] or ""
        if "\n\n> 来源：" in answer:
            answer = answer.rsplit("\n\n> 来源：", 1)[0]
        history.append({
            "role": "assistant", "content": answer,
            "source_page": meta.get("source_page"),
            "source_snippet": meta.get("source_snippet"),
            "contexts": meta.get("contexts") or [],
            "knowledge_points": meta.get("knowledge_points") or [],
        })
    st.session_state.qa_session_id = session_id
    # 会话名取最新一条记录的名字（与 list_qa_sessions 的取法口径一致）
    st.session_state.qa_session_name = (rows[-1]["session_name"] if rows else None) or "未命名会话"
    st.session_state.chat_history = history
    st.session_state.last_qa = history[-1] if len(history) >= 2 else None
    _sync_sid_param()   # 需求 3：会话切换时地址栏 sid 参数同步更新
    # 点击会话自动跳转问答页（radio 绑定 key="page"，直接写值即生效）
    st.session_state.page = "智能问答"


@st.dialog("重命名会话")
def rename_session_dialog():
    """重命名确认框：修改会话名并同步数据库与当前会话态"""
    sid = st.session_state.get("dialog_sid")
    old = st.session_state.get("dialog_sname") or ""
    new = st.text_input("会话名称", value=old, max_chars=30, key=f"rename_{sid}")
    c1, c2 = st.columns(2)
    if c1.button("保存", type="primary", use_container_width=True, disabled=not new.strip()):
        rename_qa_session(sid, new.strip(), st.session_state.user_name, st.session_state.role)
        if sid == st.session_state.qa_session_id:
            st.session_state.qa_session_name = new.strip()
        st.rerun()   # dialog 内 rerun 会自动关闭对话框
    if c2.button("取消", use_container_width=True):
        st.rerun()


@st.dialog("删除会话")
def delete_session_dialog():
    """删除确认框：级联删除该会话的全部问答记录（不可恢复）"""
    sid = st.session_state.get("dialog_sid")
    name = st.session_state.get("dialog_sname") or "未命名会话"
    st.warning(f"确定删除会话「{name}」？\n\n该会话的全部问答记录将一并删除，且不可恢复。")
    c1, c2 = st.columns(2)
    if c1.button("删除", type="primary", use_container_width=True):
        delete_qa_session(sid, st.session_state.user_name, st.session_state.role)
        # 删除的是当前打开的会话 -> 清空气泡与会话态
        if sid == st.session_state.qa_session_id:
            st.session_state.qa_session_id = None
            st.session_state.qa_session_name = None
            st.session_state.chat_history = []
            st.session_state.last_qa = None
            _sync_sid_param()   # 需求 3：会话删除后移除地址栏 sid 参数
        st.session_state.pop("dialog_sid", None)    # 弹窗用完即清，避免残留
        st.session_state.pop("dialog_sname", None)
        st.rerun()
    if c2.button("取消", use_container_width=True):
        st.session_state.pop("dialog_sid", None)
        st.session_state.pop("dialog_sname", None)
        st.rerun()


@st.dialog("删除文件")
def delete_file_dialog():
    """删除文件确认框：级联删除课件全部关联记录 + 物理原件（引用计数保护）

    弹窗所需的目标文件信息（id/文件名/关联数/物理路径）由调用方预先写入
    session_state["dialog_file"]，与 delete_session_dialog 的传参方式保持一致。
    """
    info = st.session_state.get("dialog_file") or {}
    fid = info.get("id")
    fname = info.get("filename") or ""
    st.warning(
        f"确定删除课件「{fname}」？\n\n"
        f"其问答 / 诊断 / 路径 / 评估共 {info.get('n_records', 0)} 条关联记录将一并删除，且不可恢复。"
    )
    c1, c2 = st.columns(2)
    if c1.button("删除", type="primary", use_container_width=True):
        store_path = info.get("store_path")   # 原件物理存储路径（旧记录可能为 None）
        # delete_file 强制归属校验：只删当前用户自己的课件及关联记录
        if delete_file(fid, user_name=st.session_state.user_name,
                       role=st.session_state.role):
            # 物理原件清理：仅当数据库中已无其他课件记录引用同一文件时才删除
            # （同名重复上传会产生多条记录指向同一文件，须保留原件）
            if store_path and count_files_by_store_path(store_path) == 0:
                storage.delete_original(store_path, st.session_state.user_name)
            log_event("delete_file", f"删除「{fname}」及 {info.get('n_records', 0)} 条关联记录",
                      user_name=st.session_state.user_name, role=st.session_state.role)
            if st.session_state.get("file_id") == fid:
                clear_doc_state()   # 删除的是当前课件：清空相关会话状态
            st.session_state.pop("dialog_file", None)   # 弹窗用完即清，避免残留
            st.rerun()
        else:
            st.error("删除失败：文件不存在或不属于当前用户。")
    if c2.button("取消", use_container_width=True):
        st.session_state.pop("dialog_file", None)
        st.rerun()


def _session_export_name():
    """导出文件名：取会话名并过滤文件系统非法字符"""
    name = str(st.session_state.qa_session_name or "会话").strip() or "会话"
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


def build_session_markdown(session_name, filename, history):
    """把整个会话导出为 Markdown 文本（含每轮来源标注）"""
    lines = [
        f"# 会话导出：{session_name}",
        "",
        f"> 课件：{filename or '智学 AI 学习助手'} ｜ 导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]
    for msg in history:
        if msg["role"] == "user":
            lines += ["### 我", "", msg["content"], ""]
        else:
            lines += ["### AI", "", msg["content"], ""]
            # AI 回答附来源行（与界面气泡一致的可溯源要求）
            src = build_source_line(
                msg.get("source_page"), msg.get("source_snippet"),
                (msg.get("contexts") or [{}])[0].get("pages"),
            )
            if src:
                lines += [src.strip(), ""]
    return "\n".join(lines)


def build_session_pdf(session_name, filename, history):
    """
    把整个会话导出为 PDF（fpdf2 + 系统自带中文字体，无需联网）。
    :return: PDF bytes；fpdf2 未安装或找不到中文字体时返回 None（前端降级只出 Markdown）
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    # 中文字体：逐个尝试系统字体，并做"宽度自检"——
    # 部分中文字体（如 SimHei 的 symbol cmap）字符宽度全为 0，渲染会直接报错，需提前剔除
    font_name = None
    for i, path in enumerate((r"C:\Windows\Fonts\Deng.ttf",     # 等线（Win10/11 自带，Unicode TTF）
                              r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑（ttc，新版 fpdf2 支持）
                              r"C:\Windows\Fonts\simhei.ttf",   # 黑体
                              r"C:\Windows\Fonts\simfang.ttf",  # 仿宋
                              r"C:\Windows\Fonts\simkai.ttf")): # 楷体
        if not os.path.exists(path):
            continue
        try:
            fam = f"cjk{i}"   # 每个候选用独立字体名，避免同名注册冲突
            pdf.add_font(fam, "", path)
            pdf.set_font(fam, size=11)
            if pdf.get_string_width("测试") > 0:
                font_name = fam
                break
        except Exception:
            continue
    if font_name is None:
        return None
    # 标题 + 副题
    pdf.set_font(font_name, size=16)
    pdf.multi_cell(0, 10, f"会话导出：{session_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, size=9)
    pdf.set_text_color(130)
    pdf.multi_cell(0, 6, f"课件：{filename or '智学 AI 学习助手'}    "
                         f"导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(3)
    # 逐轮正文（PDF 内置字体无 emoji 字形，用纯文字前缀区分角色）
    for msg in history:
        if msg["role"] == "user":
            pdf.set_text_color(20, 90, 170)
            pdf.set_font(font_name, size=12)
            pdf.multi_cell(0, 8, f"我：{msg['content']}", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_text_color(30)
            pdf.set_font(font_name, size=11)
            pdf.multi_cell(0, 7, f"AI：{msg['content']}", new_x="LMARGIN", new_y="NEXT")
            # 来源小字（灰色 9pt，与界面展示口径一致）
            page_txt = format_source_page(
                msg.get("source_page"),
                (msg.get("contexts") or [{}])[0].get("pages"),
            )
            snip = str(msg.get("source_snippet") or "").strip()
            if page_txt or snip:
                parts = [p for p in (page_txt, f"「{snip[:60]}」" if snip else "") if p]
                pdf.set_text_color(130)
                pdf.set_font(font_name, size=9)
                pdf.multi_cell(0, 6, "来源：" + " - ".join(parts),
                               new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.ln(3)
    return bytes(pdf.output())


@st.dialog("导出当前会话")
def export_session_dialog():
    """导出选择框：Markdown 永远可用；PDF 依赖 fpdf2（未安装时提示安装命令）"""
    name = _session_export_name()
    filename = st.session_state.doc.get("filename")
    history = st.session_state.chat_history
    # Markdown 导出（纯文本生成，零依赖）
    md = build_session_markdown(st.session_state.qa_session_name or "会话", filename, history)
    st.toast("导出文件已生成，请点击下载保存")   # 需求 1：导出成功反馈
    st.download_button(
        "下载 Markdown", data=md.encode("utf-8"),
        file_name=f"{name}_{datetime.now():%Y%m%d_%H%M%S}.md",
        mime="text/markdown", use_container_width=True,
    )
    # PDF 导出（fpdf2 可用才显示按钮）
    pdf = build_session_pdf(st.session_state.qa_session_name or "会话", filename, history)
    if pdf is None:
        st.caption("PDF 导出需要 fpdf2 库（仅首次需要安装），安装后即可使用：")
        st.code("pip install fpdf2", language="bash")
    else:
        st.download_button(
            "下载 PDF", data=pdf,
            file_name=f"{name}_{datetime.now():%Y%m%d_%H%M%S}.pdf",
            mime="application/pdf", use_container_width=True,
        )


def _handle_chat_question(question):
    """
    执行一轮完整问答：混合检索 -> 引导式回答 -> 写入对话历史与学习记录。
    提示词三合一：检索到的课件上下文 + 最近 5 轮对话历史（TutorAgent 内部组装）+ 当前提问。
    AI 回答强制绑定来源（source_page / source_snippet），气泡末尾展示来源小字。
    """
    # 开发者端开关：问答 Agent 被停用时直接提示（"重新生成"也走本函数，一处拦截全覆盖）
    if not get_agent_config()["tutor_enabled"]:
        agent_disabled_msg("「问答 Tutor Agent」")
        return
    # 轻量限流（需求 2）：单用户每分钟最多 5 次 AI 请求，超限短路（提示已由检查函数显示）
    if not ai_rate_limit_check():
        return
    with st.spinner("混合检索：向量召回 + 知识图谱匹配..."):
        result = st.session_state.retriever.retrieve(
            question, top_k=3, graph=st.session_state.doc.get("graph"),
            min_score=get_agent_config()["retrieval_min_score"],   # 置信度阈值：低相关片段不入引用
        )
    contexts = result["chunks"]   # 重排序后的 Top-3 片段（含 pages 页码信息）
    tutor = TutorAgent()
    # 限时 30s（默认值）+ 执行期间显示"AI 正在思考中..."；超时/失败 -> 友好提示，页面不崩溃
    reply = call_agent_limited(
        lambda: tutor.ask(
            question, contexts, result["knowledge_points"],
            history=st.session_state.chat_history[-10:],   # 最近 5 轮对话，支持追问理解
        ),
        agent=tutor, timeout=AGENT_CALL_TIMEOUT, spinner_text="AI 正在思考中...",
    )
    if reply:
        content = reply["content"]
        source_page = reply.get("source_page")
        source_snippet = reply.get("source_snippet")
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({
            "role": "assistant", "content": content,
            "source_page": source_page, "source_snippet": source_snippet,   # 来源（气泡小字渲染）
            "contexts": contexts, "knowledge_points": result["knowledge_points"],
        })
        st.session_state.last_qa = st.session_state.chat_history[-1]
        # ---- 多会话归属：未开话时自动新建；首条提问自动作为会话名（ChatGPT 式命名） ----
        if not st.session_state.qa_session_id:
            _new_qa_session()
        if st.session_state.qa_session_name in (None, "新对话"):
            st.session_state.qa_session_name = question.strip()[:12] or "新对话"
            # 同步该会话已落库旧记录的名字（首条占位名是"新对话"）；组不存在时更新 0 行，无害
            rename_qa_session(st.session_state.qa_session_id, st.session_state.qa_session_name,
                              st.session_state.user_name, st.session_state.role)
        # 问答记录入库：正文 + 来源引用行一起存（历史回显自带来源，零数据迁移）；
        # meta 存完整上下文（contexts/知识点/来源），供侧边栏点击会话时恢复完整气泡；
        # AI 未声明页码时用 Top-1 引用片段的页码兜底
        save_qa(ensure_file_id(), question,
                content + build_source_line(source_page, source_snippet,
                                            contexts[0].get("pages") if contexts else None),
                [c["id"] for c in contexts],
                user_name=st.session_state.user_name, role=st.session_state.role,
                session_id=st.session_state.qa_session_id,
                session_name=st.session_state.qa_session_name,
                meta={"contexts": contexts, "knowledge_points": result["knowledge_points"],
                      "source_page": source_page, "source_snippet": source_snippet})
    else:
        ai_fail_hint(tutor)   # 用户端只显示"AI服务暂时不可用/网络不稳"等安全文案


def page_chat():
    st.title("智能问答")
    st.caption("基于课件内容的多轮对话问答，回答自动标注来源片段；支持追问、重新生成与清空对话。")
    if st.session_state.retriever is None:
        empty_doc_guide()   # 需求 3：空状态引导 + 一键跳转上传页
        return

    history = st.session_state.chat_history
    # 新建会话后的空状态提示：让"新建对话"在问答页也有可见反馈
    if not history and st.session_state.qa_session_id:
        st.info("已开启新对话——输入你的第一个问题吧（首条提问后自动保存到历史会话）")

    # ---- 历史对话气泡（ChatGPT 式：用户/AI 交替，AI 气泡带来源小字与知识点折叠区） ----
    for idx, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                ctxs = msg.get("contexts") or []
                # 来源小字：AI 回答末尾标注依据页码与原文片段（AI 生成内容必须可溯源）；
                # AI 未声明页码时用 Top-1 引用片段的页码兜底
                page_txt = format_source_page(msg.get("source_page"),
                                              ctxs[0].get("pages") if ctxs else None)
                snip = str(msg.get("source_snippet") or "").strip()
                if page_txt or snip:
                    parts = [p for p in (page_txt, f"「{snip[:60]}」" if snip else "") if p]
                    st.caption("来源：" + " - ".join(parts))
                if ctxs:
                    with st.expander("引用的课件片段（含相关度）"):
                        for c in ctxs:
                            st.markdown(f"**{c['id']}**（相关度 {c['score']:.2f}）")
                            st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
                kps = msg.get("knowledge_points") or []
                if kps:
                    with st.expander("命中的知识点（向量 + 图谱混合检索）"):
                        for p in kps:
                            icon = "" if p["match_type"] == "命中" else ""
                            st.markdown(
                                f"{icon} **{p['name']}**（{p['difficulty']}）· {p['match_type']}"
                                f" —— {p['description'] or '暂无描述'}"
                            )
                # 最新一轮回答才提供"导出笔记 / 重新生成"，避免历史气泡堆积按钮
                if idx == len(history) - 1 and idx >= 1:
                    qa_for_note = {
                        "question": history[idx - 1]["content"],
                        "answer": msg["content"],
                        "source_page": msg.get("source_page"),
                        "source_snippet": msg.get("source_snippet"),
                        "contexts": ctxs,
                        "knowledge_points": kps,
                    }
                    md_note = build_qa_markdown(qa_for_note, st.session_state.doc["filename"])
                    st.download_button(
                        "导出笔记（Markdown）",
                        data=md_note.encode("utf-8"),   # 显式 UTF-8 编码，避免中文乱码
                        file_name=f"学习笔记_{st.session_state.doc['filename'] or '智学'}_{datetime.now():%Y%m%d_%H%M%S}.md",
                        mime="text/markdown",
                        key=f"dl_note_{idx}",
                    )
                    if st.button("重新生成", key=f"regen_{idx}"):
                        # 弹出最后一轮，用同一个问题重新检索作答
                        prev_question = history[idx - 1]["content"]
                        st.session_state.chat_history = history[:idx - 1]
                        _handle_chat_question(prev_question)
                        st.rerun()

    # ---- 工具行：导出当前会话 / 删除当前会话（替代旧"清空对话"，多会话下语义更明确） ----
    if history:
        c1, c2 = st.columns(2)
        if c1.button("导出本会话", use_container_width=True):
            export_session_dialog()
        if c2.button("删除当前会话", use_container_width=True):
            st.session_state.dialog_sid = st.session_state.qa_session_id
            st.session_state.dialog_sname = st.session_state.qa_session_name
            delete_session_dialog()

    # ---- 底部输入框（回车即发送） ----
    question = st.chat_input("输入你的问题，例如：这一章的核心概念是什么？")
    if question and question.strip():
        _handle_chat_question(question)
        st.rerun()


# ========== 页面 3：知识图谱 ==========
def page_graph():
    st.title("知识图谱")
    doc = st.session_state.doc
    if not doc["chunks"]:
        empty_doc_guide()   # 需求 3：空状态引导 + 一键跳转上传页
        return

    # 没有图谱时提供构建入口（空态用插图卡片呈现；失败提示由 build_graph_with_spinner 统一显示）
    if doc["graph"] is None:
        _empty_card(
            "知识图谱尚未生成",
            "点击下方按钮，AI 将根据知识点候选与课件内容生成知识点依赖关系图",
            svg="""
<svg width="128" height="100" viewBox="0 0 128 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="64" cy="50" r="38" fill="#f0f6ff"/>
  <circle cx="44" cy="36" r="9" fill="#3478f6"/>
  <circle cx="86" cy="40" r="7" fill="#7aa8f7"/>
  <circle cx="56" cy="70" r="7" fill="#7aa8f7"/>
  <circle cx="90" cy="68" r="6" fill="#c4d7fb"/>
  <path d="M52 41l30 2M48 44l6 20M84 45l4 18M60 36l20 2" stroke="#b9cffa" stroke-width="2.5"/>
</svg>""",
        )
        if st.button("构建知识图谱", type="primary"):
            if build_graph_with_spinner():
                st.toast("知识图谱已生成")
                st.rerun()
        return

    G = doc["graph"]   # networkx.DiGraph：节点=知识点（含 description/difficulty），边=先修关系
    if G.number_of_nodes() == 0:
        st.warning("图谱为空：知识点候选不足，无法构建。")
        return

    # ---- 统计信息 ----
    m1, m2, m3 = st.columns(3)
    m1.metric("知识点节点", G.number_of_nodes())
    m2.metric("先修关系", G.number_of_edges())
    diff_count = Counter(d for _, d in G.nodes(data="difficulty"))
    m3.metric("困难知识点", diff_count.get("困难", 0))

    # ---- 工具行：难度筛选器 + 导出图谱按钮 ----
    c_filter, c_export = st.columns([3, 1])
    with c_filter:
        sel_diffs = st.multiselect(
            "按难度筛选节点", ["基础", "中等", "困难"],
            default=["基础", "中等", "困难"],
            help="只显示所选难度的知识点（先修关系自动保留两端都在选中范围内的边）",
        )
    keep = [n for n, d in G.nodes(data="difficulty") if (d or "") in sel_diffs]
    G_view = G.subgraph(keep) if keep else None   # 子图：节点过滤后，边只留两端都保留的
    if G_view is None or G_view.number_of_nodes() == 0:
        st.warning("当前筛选条件下没有节点，请至少选择一个难度。")
        return

    # ---- 知识点 -> 详情映射：点击节点在右侧面板展示（描述/依赖关系/原文出处） ----
    node_sources = {}
    for name in G_view.nodes:
        name_s = str(name)
        attrs = G.nodes[name]
        # 依赖关系基于完整图谱统计：即使依赖节点被筛选隐藏，详情里仍能看到完整先修链
        prereq = sorted(str(p) for p in G.predecessors(name_s))   # 前驱 = 先修基础
        nxt = sorted(str(s) for s in G.successors(name_s))        # 后继 = 后续可学
        info = {"difficulty": str(attrs.get("difficulty") or ""),
                "desc": str(attrs.get("description") or ""),
                "prereq": prereq, "next": nxt,
                "pages": [], "text": ""}
        for c in doc["chunks"]:
            if name_s in (c.get("text") or ""):
                info["pages"] = c.get("pages") or []
                info["text"] = (c.get("text") or "")[:500]   # 截断 500 字，防 HTML 过大
                break   # 只取知识点首次出现的文本块作为出处
        node_sources[name_s] = info

    with c_export:
        # 导出当前筛选后的交互式图谱：pyvis 生成的 HTML 自带 vis.js 引用与详情面板，
        # 下载后浏览器直接打开即可使用（与页面所见一致）
        export_html = build_graph_html(G_view, height="700px", node_sources=node_sources)
        st.download_button(
            "导出图谱（HTML）", data=export_html.encode("utf-8"),
            file_name=f"知识图谱_{doc['filename'] or '智学'}_{datetime.now():%Y%m%d_%H%M%S}.html",
            mime="text/html",
            help=f"导出当前筛选视图（{G_view.number_of_nodes()}/{G.number_of_nodes()} 个节点），含节点详情面板",
        )

    # ---- 生成图谱可视化 HTML（渲染逻辑统一在 utils/graph_viz.py） ----
    components.html(build_graph_html(G_view, height="600px", node_sources=node_sources),
                    height=620, scrolling=False)
    st.caption("节点颜色=难度（绿:基础 / 橙:中等 / 红:困难），箭头=先修方向；点击节点在右侧查看详情（简介 / 依赖关系 / 课件原文出处）。基于 vis.js CDN 渲染（需联网），拖拽可调整布局。")

    # ---- 按节点度数找出核心知识点 ----
    degree = dict(G.degree())
    top5 = sorted(degree.items(), key=lambda x: -x[1])[:5]
    st.markdown("**核心知识点（关联最多）：** " + "、".join(f"{k}（{v} 条关联）" for k, v in top5))

    with st.expander("重新构建图谱"):
        if st.button("重新生成"):
            st.session_state.doc["graph"] = None
            st.rerun()


# ========== 页面 4：学习诊断 ==========
def render_diagnosis_md(report):
    """把诊断报告 JSON（DiagnosisAgent.diagnose 的返回值）渲染成 Markdown"""
    lines = [f"**得分：{report.get('score', 0)} / {report.get('total', 0)}**"]

    detail = report.get("detail") or []
    if detail:
        lines += ["", "### 逐题明细"]
        for d in detail:
            mark = "" if d["result"] == "正确" else ""
            yours = d["your_answer"] or "未作答"
            lines.append(
                f"- {mark} 第{d['index']}题「{d['knowledge_point']}」："
                f"你选 {yours}，正确答案 {d['correct_answer']}"
            )
            # 题目来源小字（出题依据的课件页码/原文片段），Markdown 斜体弱化展示
            page_txt = format_source_page(d.get("source_page"))
            snip = str(d.get("source_snippet") or "").strip()
            if page_txt or snip:
                parts = [p for p in (page_txt, f"「{snip[:60]}」" if snip else "") if p]
                lines.append(f"  *来源：{' - '.join(parts)}*")

    mastery = report.get("mastery") or []
    if mastery:
        lines += ["", "### 知识点掌握度"]
        for m in mastery:
            # 状态语义已由文字"掌握/薄弱"承载，不再用彩色图标（保持导出 Markdown 纯净）
            lines.append(f"- **{m['knowledge_point']}**：{m['status']}（答对 {m['correct']}/{m['total']} 题）")

    weak = report.get("weak_points") or []
    if weak:
        lines += ["", "### 薄弱知识点（知识域 / 历史重复度）"]
        for w in weak:
            rep = f"，历史已出现 {w['recurrence']} 次" if w.get("recurrence") else ""
            lines.append(f"- **{w['knowledge_point']}**（{w.get('domain') or '未知知识域'}{rep}）")

    errors = report.get("error_types") or []
    if errors:
        lines += ["", "### 错误类型分析"]
        for e in errors:
            lines.append(f"- **{e['knowledge_point']}**：{e['type']} —— {e['analysis']}")

    sug = report.get("suggestions") or []
    if sug:
        lines += ["", "### 学习建议"] + [f"- {s}" for s in sug]
    return "\n".join(lines)


def build_diagnosis_markdown(report, filename=None):
    """
    把诊断报告整理成可导出的 Markdown 学习报告。
    包含：用户昵称、当前文档、诊断时间、答题明细、薄弱点总结（+ 错因分析 / 学习建议）。
    """
    lines = [
        "# 学习诊断报告",
        "",
        f"- **用户昵称**：{st.session_state.user_name}",
        f"- **当前文档**：{filename or '未关联课件'}",
        f"- **诊断时间**：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- **得分**：{report.get('score', 0)} / {report.get('total', 0)}",
        "",
        "## 答题明细",
    ]
    detail = report.get("detail") or []
    if detail:
        for d in detail:
            mark = "正确" if d["result"] == "正确" else "错误"
            lines.append(
                f"{d['index']}. **{mark}**「{d['knowledge_point']}」—— "
                f"你的答案：{d['your_answer'] or '未作答'}，正确答案：{d['correct_answer']}"
            )
            if d.get("explain"):
                lines.append(f"   > 解析：{d['explain']}")
            # 题目来源（出题依据的课件页码/原文片段）
            page_txt = format_source_page(d.get("source_page"))
            snip = str(d.get("source_snippet") or "").strip()
            if page_txt or snip:
                parts = [p for p in (page_txt, f"「{snip[:60]}」" if snip else "") if p]
                lines.append(f"   > 来源：{' - '.join(parts)}")
    else:
        lines.append("（无作答明细）")

    weak = report.get("weak_points") or []
    lines += ["", "## 薄弱点总结"]
    if weak:
        lines += [
            f"- **{w['knowledge_point']}**（{w.get('domain') or '未知知识域'}"
            f"{'，历史出现 ' + str(w['recurrence']) + ' 次' if w.get('recurrence') else ''}）"
            for w in weak
        ]
    else:
        lines.append("- 全部答对，暂无薄弱知识点 ")

    errors = report.get("error_types") or []
    if errors:
        lines += ["", "## 错误类型分析"] + [
            f"- **{e['knowledge_point']}**：{e['type']} —— {e['analysis']}" for e in errors
        ]
    sug = report.get("suggestions") or []
    if sug:
        lines += ["", "## 学习建议"] + [f"- {s}" for s in sug]
    return "\n".join(lines)


def collect_eval_mastery():
    """
    聚合当前会话可用的「知识点掌握度」（0-100 整数），供评估页雷达图与导出使用。
    数据源优先级：
      1) 诊断报告 report["mastery"]：每知识点 correct/total -> 答对率；
      2) 完整评估的前后测作答（pre_/post_ widget key 仍在会话中）：
         每知识点后测正确率（未做后测时用前测）。
    两者皆无（如快速评估）返回 []，页面不渲染雷达图。
    """
    # ---- 源 1：诊断报告的掌握度明细 ----
    ms = (st.session_state.get("report") or {}).get("mastery") or []
    out = [
        {"name": m["knowledge_point"],
         "pct": round(100 * m["correct"] / m["total"]) if m.get("total") else 0}
        for m in ms if m.get("knowledge_point")
    ]
    if out:
        return out

    # ---- 源 2：完整评估前后测（选项形如 "A. xxx"，取首字母比对） ----
    pre_quiz = st.session_state.get("pre_quiz") or []
    post_quiz = st.session_state.get("post_quiz") or []
    quiz = post_quiz or pre_quiz          # 优先后测：反映学习后的真实水平
    if not quiz:
        return []
    prefix = "post" if post_quiz else "pre"
    agg = {}   # 知识点 -> [答对数, 总题数]
    for i, q in enumerate(quiz):
        kp = q.get("knowledge_point") or "未命名知识点"
        yours = str(st.session_state.get(f"{prefix}_{i}", "") or "").strip().upper()[:1]
        ok = bool(yours) and yours == str(q.get("answer", "")).strip().upper()[:1]
        c = agg.setdefault(kp, [0, 0])
        c[0] += 1 if ok else 0
        c[1] += 1
    return [{"name": kp, "pct": round(100 * c0 / c1)} for kp, (c0, c1) in agg.items()]


def build_eval_markdown(pre, post, full, alg, report_text, mastery=None):
    """
    把评估结果整理成可导出的 Markdown 报告。
    结构：用户 / 文档 / 时间 / 前后测 / ALG / 知识点掌握度表（可选）/ 报告全文。
    """
    # Hake 增益等级：>=0.7 高 / 0.3~0.7 中 / <0.3 低
    level = "高增益" if alg >= 0.7 else ("中等增益" if alg >= 0.3 else "低增益")
    lines = [
        "# 学习评估报告",
        "",
        f"- **用户昵称**：{st.session_state.user_name}",
        f"- **当前文档**：{st.session_state.doc['filename'] or '未关联课件'}",
        f"- **评估时间**：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"- **前测得分**：{pre} / {full}",
        f"- **后测得分**：{post} / {full}",
        f"- **学习增益 ALG**：{alg:.2f}（{level}）",
    ]
    # 知识点掌握度表（有数据时附在报告全文之前）
    if mastery:
        lines += ["", "## 知识点掌握度", "", "| 知识点 | 掌握度 |", "| --- | --- |"]
        lines += [f"| {m['name']} | {m['pct']}% |" for m in mastery]
    lines += ["", "## 评估报告", "", report_text or ""]
    return "\n".join(lines)


def page_diagnosis():
    st.title("学习诊断")
    st.caption("AI 从知识图谱中挑选核心知识点出选择题，作答后生成结构化诊断报告，可导出 Markdown 学习报告。")
    doc = st.session_state.doc
    if not doc["chunks"]:
        empty_doc_guide()   # 需求 3：空状态引导 + 一键跳转上传页
        return

    # ---- 阶段一：出题配置（数量 + 难度）与生成 ----
    if st.session_state.quiz is None:
        st.info("AI 将按知识图谱节点度数挑选核心知识点，各生成 1 道选择题。")
        c1, c2 = st.columns(2)
        # 出题数量默认值来自开发者端"Agent 配置"页（管理员改配置后这里实时生效）
        n_questions = c1.select_slider("出题数量", options=[3, 5, 10],
                                       value=get_agent_config()["diagnosis_n_questions"])
        difficulty = c2.radio(
            "出题难度", ["基础", "进阶", "综合"], horizontal=True,
            help="基础：概念辨认｜进阶：应用分析｜综合：混合情境、跨知识点",
        )
        if st.button("生成诊断题目", type="primary"):
            if not get_agent_config()["diagnosis_enabled"]:
                agent_disabled_msg("「诊断 Agent」")
                return
            if not ai_rate_limit_check():   # 轻量限流（需求 2）：出题计入 AI 请求配额
                return
            graph = doc.get("graph")
            if graph is None:   # 出题依赖核心知识点排名，需先有图谱
                with st.spinner("首次诊断：先构建知识图谱..."):
                    ok = build_graph_with_spinner()   # 失败提示已由其内部统一显示
                graph = st.session_state.doc["graph"] if ok else None
            if graph is not None:
                diag_agent = DiagnosisAgent()
                # 任务追踪：登记出题任务（pending -> processing -> success/failed）
                task_id = create_task("diagnosis_quiz",
                                      st.session_state.user_name, st.session_state.role)
                update_task_status(task_id, "processing")
                # 限时 90s（生成长 JSON 耗时长）：出题失败/超时走降级分支，不清历史、不崩溃
                quiz = call_agent_limited(
                    lambda: diag_agent.get_questions(graph, n=n_questions, difficulty=difficulty,
                                                     chunks=doc["chunks"]),   # 传入原文块：出题绑定来源
                    agent=diag_agent, timeout=90,
                    spinner_text=f"AI 正在生成 {n_questions} 道「{difficulty}」题目...")
                if quiz:
                    update_task_status(task_id, "success")
                    st.session_state.quiz = quiz
                    st.session_state.report = None
                    st.session_state.path = None
                    reset_eval_state()
                    st.toast(f"已生成 {n_questions} 道诊断题目")   # 需求 1
                    st.rerun()   # 重新渲染页面，立即显示题目
                else:
                    # 任务追踪：出题失败/超时，错误码映射文案入 tasks.error_msg
                    update_task_status(task_id, "failed",
                                       (diag_agent.last_error or {}).get("message", "出题失败"))
                    # 局部降级（需求 2）：出题失败提示友好文案，历史记录完好，引导去其他页面
                    st.warning(
                        "出题服务暂时繁忙，已跳过本次诊断。你的历史学习记录完好，"
                        "可先前往「智能问答」或「知识图谱」继续学习，稍后再回来出题。"
                    )
        return

    quiz = st.session_state.quiz

    # ---- 阶段二：作答（报告未生成时才渲染作答区；提交前严禁显示任何对错） ----
    if st.session_state.report is None:
        st.subheader("请作答")
        render_quiz_answers(quiz, "quiz")   # 渲染层数据已剔除 answer/explain，防泄漏
        if st.button("提交答案", type="primary"):
            if not get_agent_config()["diagnosis_enabled"]:
                agent_disabled_msg("「诊断 Agent」")
                return
            if not ai_rate_limit_check():   # 轻量限流（需求 2）：诊断报告计入 AI 请求配额
                return
            answers = collect_answers(quiz, "quiz")   # 取首字母还原选项字母
            diag_agent = DiagnosisAgent()
            # 任务追踪：登记诊断报告任务
            task_id = create_task("diagnosis_report",
                                  st.session_state.user_name, st.session_state.role)
            update_task_status(task_id, "processing")
            # 限时 90s + 降级：报告生成失败仅提示重试，作答记录仍在页面上不丢失
            report = call_agent_limited(
                lambda: diag_agent.diagnose(
                    answers, quiz, graph=doc.get("graph"),
                    history=get_history(st.session_state.user_name, st.session_state.role)   # 只统计当前用户的历史薄弱点
                ),
                agent=diag_agent, timeout=90, spinner_text="AI 正在分析错误原因并生成诊断报告...")
            if report:
                update_task_status(task_id, "success")
                st.session_state.report = report
                st.toast("诊断报告已生成")   # 需求 1：报告生成 Toast
                # 诊断记录入库（题目/答案/得分/报告），供后续诊断统计"薄弱知识点重复度"
                save_diagnosis(ensure_file_id(), quiz, answers, report["score"], report,
                               user_name=st.session_state.user_name, role=st.session_state.role)
                st.rerun()
            else:
                update_task_status(task_id, "failed",
                                   (diag_agent.last_error or {}).get("message", "诊断报告生成失败"))
                ai_fail_hint(diag_agent, fallback="诊断失败，请重试。")
        return

    # ---- 阶段三：报告（JSON -> Markdown 渲染 + 掌握度 + 导出） ----
    st.divider()
    st.markdown("### 诊断报告")
    report = st.session_state.report
    # ---- 逐题对错柱状图：答对=1，答错=0，一眼看出哪几题失分 ----
    detail = report.get("detail") or []
    if detail:
        chart_data = {f"第{d['index']}题\n{d['knowledge_point']}": (1 if d["result"] == "正确" else 0) for d in detail}
        st.bar_chart(chart_data, height=260)
        st.caption("纵轴：1 = 正确，0 = 错误；横轴为题号与考点")
    else:
        st.caption("暂无逐题对错数据。")   # 图表空态占位（需求 3）
    st.markdown(render_diagnosis_md(report))

    # ---- 导出学习报告（Markdown：昵称 / 文档 / 时间 / 明细 / 薄弱点总结） ----
    md_report = build_diagnosis_markdown(report, st.session_state.doc["filename"])
    st.download_button(
        "导出学习报告（Markdown）",
        data=md_report.encode("utf-8"),
        file_name=f"诊断报告_{st.session_state.doc['filename'] or '智学'}_{datetime.now():%Y%m%d_%H%M%S}.md",
        mime="text/markdown",
    )

    # ---- 回看作答（只显示自己的选择，对错结论以报告为准） ----
    with st.expander("查看我的作答"):
        for i, q in enumerate(quiz):
            choice = (st.session_state.get(f"quiz_{i}", "") or "").strip() or "未作答"
            st.markdown(f"- 第{i + 1}题「{q.get('knowledge_point', '')}」：你选择了 **{choice}**")

    st.caption("可前往「路径推荐」生成学习路径，或到「学习评估」查看学习增益。")

    if st.button("重新测试"):
        st.session_state.quiz = None
        st.session_state.report = None
        st.session_state.path = None
        reset_eval_state()
        for i in range(len(quiz)):
            st.session_state.pop(f"quiz_{i}", None)
        st.rerun()


# ========== 页面 5：路径推荐 ==========
def page_path():
    st.title("路径推荐")
    st.caption("根据诊断报告的薄弱知识点与知识图谱先修依赖，用拓扑排序生成有序学习路径。")
    doc = st.session_state.doc
    if not doc["chunks"]:
        empty_doc_guide()   # 需求 3：空状态引导 + 一键跳转上传页
        return
    if not st.session_state.report and not st.session_state.path:
        st.info("还没有诊断报告。请先到「学习诊断」页完成一次测验，路径会更有针对性。")
        return

    if st.session_state.path is None:
        if st.button("生成学习路径", type="primary"):
            if not get_agent_config()["path_enabled"]:
                agent_disabled_msg("「路径 Agent」")
                return
            # 纯本地拓扑排序，瞬时完成：先补前置知识，再攻薄弱点（加 spinner 统一加载体验）
            # 任务追踪：登记路径生成任务（本地计算，正常必成功；异常时标记 failed）
            task_id = create_task("path_generate",
                                  st.session_state.user_name, st.session_state.role)
            update_task_status(task_id, "processing")
            with st.spinner("正在根据薄弱点与先修依赖生成学习路径..."):
                try:
                    path = PathAgent().generate(st.session_state.report, graph=doc["graph"])
                except Exception as e:
                    update_task_status(task_id, "failed", f"路径生成异常：{e}")
                    error_logger.error("PathAgent.generate 异常", exc_info=True)
                    st.error("学习路径生成出现问题，请稍后重试。")
                    return
            if path:
                update_task_status(task_id, "success")
                st.session_state.path = path
                st.toast("学习路径已生成")   # 需求 1
                save_path(ensure_file_id(), path,
                          user_name=st.session_state.user_name, role=st.session_state.role)   # 路径列表自动转 JSON 入库
                st.rerun()
            else:
                # 任务本身执行成功，只是结果为空（全部答对无需路径）：标记 success 不算失败
                update_task_status(task_id, "success")
                st.warning("诊断报告中没有薄弱知识点（可能全部答对了），无需生成路径。")
    else:
        # 渲染有序路径列表：编号 + 知识点 + 学习理由
        for step in st.session_state.path:
            st.markdown(f"**{step['order']}. {step['knowledge_point']}**")
            st.caption(step["reason"])
        if st.button("重新生成路径"):
            st.session_state.path = None
            st.rerun()


# ========== 页面 6：学习评估 ==========
def page_eval():
    st.title("学习评估")
    st.caption("前测 → 复习 → 后测，用归一化学习增益（ALG）量化你的进步幅度。")
    doc = st.session_state.doc
    if not doc["chunks"]:
        empty_doc_guide()   # 需求 3：空状态引导 + 一键跳转上传页
        return

    # ---- 已有评估结果：展示增益指标、掌握度雷达图、报告与导出（快速/完整评估共用） ----
    ev = st.session_state.evaluation
    if ev:
        col1, col2, col3 = st.columns(3)
        col1.metric("前测得分", f"{ev['pre_score']}/{ev['full_score']}")
        col2.metric("后测得分", f"{ev['post_score']}/{ev['full_score']}")
        col3.metric("学习增益 ALG", f"{ev['alg']:.2f}")
        st.divider()

        # ---- 知识点掌握度雷达图（plotly；知识点 < 3 个时降级为条形图） ----
        # 数据来源：诊断报告的掌握度明细，或完整评估的前后测作答（collect_eval_mastery 内部择优）
        mastery = collect_eval_mastery()
        fig = None   # 掌握度数据不足时为 None（导出区据此隐藏 PNG 按钮）
        if mastery:
            st.markdown("##### 知识点掌握度雷达")
            fig = radar_fig(mastery)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            else:
                st.bar_chart(bar_chart_data(mastery), height=260)
                st.caption("知识点不足 3 个，暂以条形图展示（雷达图至少需要 3 个维度）。")
        else:
            # 图表空态占位（需求 3）：没有掌握度数据时不留空白
            st.caption("暂无掌握度数据——完成一次「学习诊断」后，这里会出现掌握度雷达图。")

        st.markdown(ev["report"])

        # ---- 导出区：图片（SVG 报告卡为主，kaleido 可用时附 PNG）+ Markdown ----
        eval_meta = {
            "user": st.session_state.user_name,
            "filename": st.session_state.doc["filename"] or "未关联课件",
            "pre": ev["pre_score"], "post": ev["post_score"], "full": ev["full_score"],
            "alg": ev["alg"], "time": f"{datetime.now():%Y-%m-%d %H:%M}",
        }
        ec1, ec2 = st.columns(2)
        # 图片导出：零依赖 SVG 报告卡（雷达图 + 分数指标，矢量图可直接插入 Word/PPT）
        svg_card = radar_report_svg(mastery, eval_meta)
        fn_base = f"{st.session_state.doc['filename'] or '智学'}_{datetime.now():%Y%m%d_%H%M%S}"
        ec1.download_button(
            "导出报告卡（SVG 图片）",
            data=svg_card.encode("utf-8"),
            file_name=f"评估报告卡_{fn_base}.svg",
            mime="image/svg+xml",
            help="SVG 为矢量图片格式，浏览器可直接打开，Word / PPT 可直接插入",
        )
        # PNG 导出（可选增强）：安装 kaleido 后自动提供雷达图 PNG；失败则静默不显示
        if fig is not None:
            png_bytes = radar_png_bytes(fig)
            if png_bytes:
                ec1.download_button(
                    "导出雷达图（PNG）",
                    data=png_bytes,
                    file_name=f"掌握度雷达_{fn_base}.png",
                    mime="image/png",
                )
        # Markdown 导出：含掌握度表格 + 报告全文
        md_eval = build_eval_markdown(ev["pre_score"], ev["post_score"], ev["full_score"],
                                      ev["alg"], ev["report"], mastery)
        ec2.download_button(
            "导出评估报告（Markdown）",
            data=md_eval.encode("utf-8"),
            file_name=f"评估报告_{fn_base}.md",
            mime="text/markdown",
        )
        if st.button("重新评估"):
            reset_eval_state()
            st.rerun()
        return

    # ---- 双模式：快速评估（手输分数，零 API 成本） / 完整评估（前后测作答） ----
    tab_quick, tab_quiz = st.tabs(["快速评估（输入分数）", "完整评估（前后测作答）"])

    with tab_quick:
        st.caption("直接输入前测 / 后测分数，立即计算学习增益：ALG = (后测 − 前测) / (满分 − 前测)")
        c1, c2, c3 = st.columns(3)
        pre = c1.number_input("前测分数", min_value=0, max_value=200, value=40, key="quick_pre")
        post = c2.number_input("后测分数", min_value=0, max_value=200, value=70, key="quick_post")
        full = c3.number_input("满分", min_value=1, max_value=200, value=100, key="quick_full")
        if st.button("计算学习增益", type="primary"):
            if not get_agent_config()["eval_enabled"]:
                agent_disabled_msg("「评估 Agent」")
                return
            alg = EvalAgent.calc_alg(pre, post, full)
            # Hake 增益等级：>=0.7 高 / 0.3~0.7 中 / <0.3 低
            if alg >= 0.7:
                level, tip = "高增益", "学习策略非常有效，继续保持！"
            elif alg >= 0.3:
                level, tip = "中等增益", "有一定进步，建议针对薄弱点继续巩固。"
            else:
                level, tip = "低增益", "进步有限，建议回到「路径推荐」重新复习薄弱知识点。"
            local_report = (
                f"**前测 {pre} → 后测 {post}（满分 {full}）**\n\n"
                f"**学习增益 ALG：{alg:.2f}（{level}）**\n\n{tip}"
            )
            # 入库（report 列存报告文本，供历史页回溯）+ 写入 session_state
            save_eval(ensure_file_id(), pre, post, alg, local_report,
                      user_name=st.session_state.user_name, role=st.session_state.role)
            st.session_state.evaluation = {
                "pre_score": pre, "post_score": post, "full_score": full,
                "alg": alg, "report": local_report,
            }
            st.rerun()   # 统一跳到顶部结果展示区（指标卡 + 报告 + 导出按钮）

    with tab_quiz:
        _full_eval_flow(doc)


def _full_eval_flow(doc):
    """完整评估流程：前测摸底 -> 复习 -> 后测 -> AI 增益报告（结果写入 session_state.evaluation）"""
    # ---- 阶段一：前测（摸底） ----
    if st.session_state.pre_score is None:
        if st.session_state.pre_quiz is None:
            st.info("先做一次前测摸底；复习后再做后测，系统将计算你的学习增益。")
            if st.button("开始前测", type="primary"):
                if not get_agent_config()["eval_enabled"]:
                    agent_disabled_msg("「评估 Agent」")
                    return
                if not ai_rate_limit_check():   # 轻量限流（需求 2）：前测出题计入配额
                    return
                eval_agent = EvalAgent()
                # 限时 90s + 降级：前测出题失败仅提示重试，页面状态不变
                quiz = call_agent_limited(
                    lambda: eval_agent.pre_test(
                        graph=doc.get("graph"),
                        knowledge_candidates=doc["knowledge_candidates"],
                    ),
                    agent=eval_agent, timeout=90, spinner_text="AI 正在生成前测题目...")
                if quiz:
                    st.session_state.pre_quiz = quiz
                    st.rerun()
                else:
                    ai_fail_hint(eval_agent, fallback="前测题目生成失败，请重试。")
        else:
            st.subheader("前测")
            render_quiz_answers(st.session_state.pre_quiz, "pre")
            if st.button("提交前测答案", type="primary"):
                st.session_state.pre_score = EvalAgent().grade(
                    st.session_state.pre_quiz, collect_answers(st.session_state.pre_quiz, "pre")
                )
                st.rerun()
        return

    # ---- 阶段二：后测（与前测平行的卷子） ----
    if st.session_state.post_score is None:
        if st.session_state.post_quiz is None:
            st.info(
                f"前测得分 {st.session_state.pre_score}。"
                "建议先到「路径推荐」复习薄弱知识点，完成后再回来做后测。"
            )
            if st.button("开始后测", type="primary"):
                if not get_agent_config()["eval_enabled"]:
                    agent_disabled_msg("「评估 Agent」")
                    return
                if not ai_rate_limit_check():   # 轻量限流（需求 2）：后测出题计入配额
                    return
                eval_agent = EvalAgent()
                # 限时 90s + 降级：后测出题失败仅提示重试，前测成绩不受影响
                quiz = call_agent_limited(
                    lambda: eval_agent.post_test(
                        pre_quiz=st.session_state.pre_quiz,
                        graph=doc.get("graph"),
                        knowledge_candidates=doc["knowledge_candidates"],
                    ),
                    agent=eval_agent, timeout=90, spinner_text="AI 正在生成后测题目（与前测考点相同、题面不同）...")
                if quiz:
                    st.session_state.post_quiz = quiz
                    st.rerun()
                else:
                    ai_fail_hint(eval_agent, fallback="后测题目生成失败，请重试。")
        else:
            st.subheader("后测")
            render_quiz_answers(st.session_state.post_quiz, "post")
            if st.button("提交后测答案", type="primary"):
                st.session_state.post_score = EvalAgent().grade(
                    st.session_state.post_quiz, collect_answers(st.session_state.post_quiz, "post")
                )
                st.rerun()
        return

    # ---- 阶段三：计算增益并生成评估报告（Agent 内部写入 SQLite） ----
    if not get_agent_config()["eval_enabled"]:
        agent_disabled_msg("「评估 Agent」")
        return
    if not ai_rate_limit_check():   # 轻量限流（需求 2）：评估报告计入配额（本页渲染时自动触发）
        return
    eval_agent = EvalAgent()
    # 限时 90s + 降级：报告生成失败仅提示重试，前后测成绩保留可重试
    data = call_agent_limited(
        lambda: eval_agent.generate_report(
            pre_score=st.session_state.pre_score,
            post_score=st.session_state.post_score,
            full_score=len(st.session_state.pre_quiz),   # 每题 1 分
            file_id=ensure_file_id(),                    # 评估结果由 Agent 内部入库
            pre_quiz=st.session_state.pre_quiz,
            post_quiz=st.session_state.post_quiz,
            user_name=st.session_state.user_name,
        ),
        agent=eval_agent, timeout=90, spinner_text="AI 正在生成学习增益评估报告...")
    if data:
        st.session_state.evaluation = data
        st.toast("评估报告已生成")   # 需求 1：报告生成 Toast
        st.rerun()
    else:
        ai_fail_hint(eval_agent, fallback="评估报告生成失败，请重试。")


# ========== 页面 7：历史记录 ==========
def page_history():
    """从 SQLite 读取全部学习记录：类型筛选 + 展开回溯完整内容"""
    st.title("历史记录")
    history = get_history(st.session_state.user_name, st.session_state.role)   # 只看当前用户的学习足迹
    if not history:
        st.info("暂无学习记录，去上传第一份课件吧！上传解析、问答、诊断后记录会自动出现在这里。")
        return

    # ---- 顶部统计概览 ----
    m1, m2, m3 = st.columns(3)
    m1.metric("记录总数", len(history))
    m2.metric("涉及课件", len({r["filename"] for r in history}))
    m3.metric("最近记录", history[0]["time"])

    # ---- 类型筛选 ----
    type_choice = st.radio(
        "按类型筛选",
        ["全部"] + list(TYPE_LABELS.keys()),
        format_func=lambda t: "全部" if t == "全部" else f"{TYPE_LABELS[t]} {t}",
        horizontal=True,
    )
    rows = history if type_choice == "全部" else [r for r in history if r["type"] == type_choice]
    if not rows:
        st.caption("该类型暂无记录。")
        return

    # ---- 逐条展开回溯完整内容（复用侧边栏的详情渲染逻辑） ----
    for r in rows:
        with st.expander(f"{TYPE_LABELS.get(r['type'], '?')} · {r['filename']} · {r['time']}"):
            render_history_detail(r)


# ========== 页面 8：用量信息（API 消耗记录与可视化） ==========
def page_usage():
    """
    展示 API 消耗情况：数据来自 api_usage 表（BaseAgent.chat 每次成功调用后写入）。
    展示当前用户自己的 API 消耗（仅本账号数据，不含余额与他人信息）：
    中间：消费金额 / API请求次数 / Tokens 三个指标卡片；
    底部：按日期统计的消费折线图。全部数据按当前登录昵称+角色过滤。
    """
    st.title("我的用量")
    st.caption(
        "记录你在本平台的 AI 调用消耗。计费口径：输入 1 元/百万 tokens、输出 2 元/百万 tokens，"
        "费用为本地估算值。"
    )
    summary = get_api_usage_summary(st.session_state.user_name, role=st.session_state.role)

    # ---- 中间：三个核心指标卡片（仅当前用户自己的数据） ----
    c1, c2, c3 = st.columns(3)
    c1.metric("消费金额", f"¥{summary['total_cost']:.4f}")
    c2.metric("API请求次数", f"{summary['requests']:,}")
    c3.metric("Tokens", f"{summary['total_tokens']:,}")

    st.divider()

    # ---- 底部：按日期统计的消费折线图 ----
    daily = get_api_usage_daily(st.session_state.user_name, role=st.session_state.role)
    if daily:
        st.subheader("每日消费趋势")
        st.line_chart(daily, height=280)
        st.caption("纵轴：当日估算消费（元）；横轴：日期。数据按当前用户过滤。")
    else:
        st.info("暂无 API 消耗记录——完成一次问答 / 诊断 / 评估后，这里会出现你的消费曲线。")


# ========== 页面 9：个人中心（学生端：账号信息 + 用量 + 存储空间） ==========
def _stat_card(label, value, help_text=None):
    """
    自定义数据卡（替代 st.metric）：小字标签 + 大字数值，长文本自动换行不截断。
      - st.metric 在窄列里会对长值（时间戳/路径）强制显示 "..."，原生无法关闭；
      - 本卡片用 word-wrap:break-word + overflow-wrap:anywhere 允许任意位置断行，
        值再长也只是换行，绝不出现省略号；
      - 安全约定：进入 unsafe_allow_html 渲染前对 label/value/help 全部 html.escape；
      - help_text 渲染为 title 悬停提示（对应 st.metric 的 help 参数）。
    """
    label_html = html.escape(str(label))
    value_html = html.escape(str(value))
    title_attr = f' title="{html.escape(str(help_text))}"' if help_text else ""
    return (
        f'<div{title_attr} style="word-wrap:break-word;overflow-wrap:anywhere;">'
        f'<div style="font-size:12px;color:#7F8C8D;margin-bottom:2px;">{label_html}</div>'
        f'<div style="font-size:18px;font-weight:bold;color:#2C3E50;'
        f'word-wrap:break-word;overflow-wrap:anywhere;line-height:1.35;">{value_html}</div>'
        f'</div>'
    )


def page_profile():
    """
    个人中心：当前账号的资料与资源占用一览（仅本人数据）：
      - 账号资料：注册时间 / 最后登录 / 账号状态（users 表）；
      - 用量统计：总 Tokens / API 调用次数 / 预估费用（api_usage 按 user_name 过滤）；
      - 存储空间：data/{user_name}/ 专属目录的文件数与硬盘占用（utils/storage）；
      三组指标统一用 _stat_card 自定义 HTML 渲染（st.metric 会截断长值）。
    """
    st.title("个人中心")
    st.caption("你的账号资料、AI 用量与专属知识库存储空间一览（仅展示本账号数据）")

    # ---- 账号资料（users 表：get_user 按昵称取本人记录） ----
    user = get_user(st.session_state.user_name) or {}
    st.subheader(f"{ROLE_BADGE.get(st.session_state.role, st.session_state.role)} "
                 f"**{st.session_state.user_name}**")
    # 时间只显示日期部分（YYYY-MM-DD），窄列也不截断
    a1, a2, a3 = st.columns(3)
    a1.markdown(_stat_card("注册时间", (user.get("created_at") or "-")[:10]),
                unsafe_allow_html=True)
    a2.markdown(_stat_card("最后登录", (user.get("last_login_at") or "-")[:10]),
                unsafe_allow_html=True)
    a3.markdown(_stat_card("账号状态", "正常" if not user.get("is_disabled") else "已禁用"),
                unsafe_allow_html=True)

    st.divider()

    # ---- 用量统计（api_usage 表按 user_name+role 双过滤，只统计本人） ----
    st.subheader("AI 用量统计")
    summary = get_api_usage_summary(st.session_state.user_name, role=st.session_state.role)
    u1, u2, u3 = st.columns(3)
    u1.markdown(_stat_card("总 Tokens 消耗", f"{summary['total_tokens']:,}"),
                unsafe_allow_html=True)
    u2.markdown(_stat_card("API 调用次数", f"{summary['requests']:,}"),
                unsafe_allow_html=True)
    u3.markdown(_stat_card("预估费用", f"¥{summary['total_cost']:.4f}",
                           help_text="计费口径：输入 1 元/百万 tokens、输出 2 元/百万 tokens（本地估算）"),
                unsafe_allow_html=True)

    st.divider()

    # ---- 存储空间统计（data/{user_name}/ 专属目录，物理隔离下的本人占用） ----
    st.subheader("专属知识库存储")
    size_bytes = storage.dir_size_bytes(st.session_state.user_name)
    size_mb = size_bytes / (1024 * 1024)
    n_files = len(storage.list_user_files(st.session_state.user_name))
    s1, s2, s3 = st.columns(3)
    s1.markdown(_stat_card("已占用空间", f"{size_mb:.2f} MB"), unsafe_allow_html=True)
    s2.markdown(_stat_card("课件原件数", f"{n_files} 个"), unsafe_allow_html=True)
    s3.markdown(_stat_card("存储位置", f"data/{st.session_state.user_name}/",
                           help_text="你的课件原件保存在服务器上的个人专属目录，与其他用户物理隔离"),
                unsafe_allow_html=True)

    # ---- 本人的每日消费趋势（与"我的用量"页同源，方便一站式查看） ----
    daily = get_api_usage_daily(st.session_state.user_name, role=st.session_state.role)
    if daily:
        st.subheader("我的每日消费趋势")
        st.line_chart(daily, height=240)
    else:
        st.info("暂无 API 消耗记录——完成一次问答后，这里会出现你的消费趋势。")


# ========== 登录页（左右分栏 · 现代 SaaS 风格） ==========
# 布局说明（飞书/抖音 B 端风格 · 苹果青 + 奶酪色 · 悬浮白卡版）：
#   全局背景 = 左上奶酪色 → 右下浅苹果青的线性渐变（.stApp 注入）；
#   左半 60% 品牌区背景透明（呈现奶酪色渐变），内容整体下移与右卡中轴对齐；
#   右半 40% 表单区 = 半透明浅奶酪色底 + 左侧极细分隔竖线，顶部 15vh 起悬浮一张
#   纯白卡片（圆角 16px、苹果青柔影、限宽 400px）——卡片锚定在稳定存在的
#   stLayoutWrapper > stVerticalBlock 结构上（不依赖会消失的 BorderWrapper）；
#   内部：Logo → 登录方式切换 → 44px 无边框输入框 → 协议复选框 → 48px 苹果青按钮；
#   "阅读协议" expander 放在卡片外、卡片正下方。
#   专属样式只随登录页渲染注入，登录成功后自动卸载，不影响内页布局。
# ========== 登录页（飞书/抖音 B 端 · 50/50 分栏 · 苹果青+奶酪色） ==========
# 布局说明（标准三段式，全部使用最简单的 padding/border/margin，无 position:absolute）：
#   左半 50% —— 奶酪色→浅苹果青渐变品牌区：花形图标 / 48px 主标题 / 英文副标 / 简介
#               垂直居中（margin:auto 实现），隐私声明用 margin-top:auto 固定底部居中；
#   右半 50% —— 极浅灰 #F9FAFB 表单区，白卡片（420px/圆角20/内边距40/苹果青柔影）用
#               margin:auto 垂直水平双居中（空间不足时自动贴顶可滚，绝不截断）；
#   卡片锚定 stLayoutWrapper > stVerticalBlock（稳定结构，不依赖会消失的 BorderWrapper）。
#   专属样式只随登录页渲染注入，登录成功后自动卸载，不影响内页布局。
LOGIN_PAGE_CSS = """
<style>
    /* ---- 全屏化：解除主容器限宽与内边距（红线：去掉默认上下 padding） ---- */
    .block-container { max-width: 100vw !important; padding: 0 !important; }

    /* ---- 两栏行：50/50 分栏，min-height 100vh（内容超出时可伸长，绝不截断） ----
       :has(#brand-main) 精确锁定登录页这一行，不波及内页任何列布局 */
    [data-testid="stHorizontalBlock"]:has(#brand-main) {
        min-height: 100vh; margin: 0 !important;
        gap: 0 !important; flex-wrap: nowrap !important; align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child,
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:last-child {
        min-height: 100vh; padding: 0 !important;
    }
    /* 右表单区：极浅灰底 + 纵向 flex；白卡悬空居中于右半部分（水平垂直双居中） */
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:last-child {
        background: #F9FAFB; display: flex; flex-direction: column;
        justify-content: center; align-items: center;
    }
    /* 白卡片外层块（列内块）：限宽 420px。
       flex:0 0 auto + height:auto 覆盖 Streamlit 默认的 flex-grow:1 / height:100%——
       否则块被拉伸满列，justify-content:center 无剩余空间可分，垂直居中失效 */
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:last-child
    > div[data-testid="stVerticalBlock"] {
        flex: 0 0 auto !important;
        height: auto !important;
        margin: auto !important; width: 100%; max-width: 420px;
    }
    /* 左品牌列：奶酪→浅苹果青渐变 + 纵向 flex——主内容块(80vh)垂直居中于上部，
       声明块 margin-top:auto 推到左栏最底部；品牌内容布局由自绘 HTML 内联样式负责 */
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child {
        background: linear-gradient(135deg, #FBF1D7 0%, #E8F3E0 100%);
        display: flex; flex-direction: column;
    }
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child
    [data-testid="stMarkdownContainer"] { padding: 0 !important; }
    /* stMarkdownContainer 水平居中（其内的 #brand-main 品牌模块跟随居中） */
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child
    > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]
    > div[data-testid="stMarkdown"] > [data-testid="stMarkdownContainer"] {
        margin-left: auto !important; margin-right: auto !important;
    }

    /* ---- 品牌主内容块：flex 双居中（布局必须在 style 块内——内联会被 Streamlit 剥离） ---- */
    #brand-main {
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        text-align: center; height: 80vh; padding-top: 6vh;   /* 顶部 6vh：整体略微下移 */
    }
    #brand-main img {
        width: 280px; margin-bottom: 30px;
        filter: drop-shadow(0 14px 28px rgba(115, 174, 82, .30));
    }
    #brand-main .bp-title {
        font-size: 42px; font-weight: 800; letter-spacing: 2px;
        color: #2C3E50; line-height: 1.2; margin-bottom: 20px;
    }
    #brand-main .bp-en {
        font-size: 14px; font-weight: 600; letter-spacing: 2px;
        color: #73AE52; text-transform: uppercase; margin-bottom: 20px;
    }
    #brand-main .bp-sub { font-size: 16px; color: #7F8C8D; font-weight: 300; }
    /* 三个功能卡片：半透明白底、圆角 12px、内边距 16px、卡间距 16px，宽度随 #brand-main */
    #brand-main .bp-cards {
        display: flex; flex-direction: column; gap: 16px;
        width: 100%; margin-top: 24px;
    }
    #brand-main .bp-card {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px; padding: 16px;
        text-align: left;
    }
    #brand-main .bp-card-title {
        font-size: 16px; font-weight: 700; color: #2C3E50; margin-bottom: 4px;
    }
    #brand-main .bp-card-desc {
        font-size: 13px; color: #7F8C8D; line-height: 1.5;
    }
    /* 底部声明：贴底居中小字 */
    #brand-foot {
        text-align: center; width: 100%; font-size: 12px;
        color: #A0A0A0; padding-bottom: 20px;
    }
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child
    > div[data-testid="stVerticalBlock"] {
        display: flex !important; flex-direction: column !important;
        align-items: center !important;   /* 水平居中所有孩子（花朵/文字组） */
        min-height: 100vh !important; gap: 0 !important;
    }
    /* ec（stElementContainer）自身也纵向 flex 居中：其内的品牌模块/声明水平居中 */
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child
    > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
        display: flex !important; flex-direction: column !important;
        justify-content: center !important; align-items: center !important;
    }
    [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child
    > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(#brand-foot) {
        margin-top: auto !important;   /* 声明推到左栏最底部 */
    }
    /* 窄屏横幅：主块高度自适应，右列保留 60vh 让白卡悬空居中 */
    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child
        > div[data-testid="stVerticalBlock"] { min-height: 0 !important; }
        #brand-main {
            height: auto !important; min-height: 0 !important; padding: 32px 24px !important;
        }
        #brand-main img { width: 200px !important; margin-bottom: 20px !important; }
        [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:last-child {
            min-height: 60vh !important;
            padding: 32px 16px 24px !important;
        }
    }

    /* ---- 悬浮白卡片：锚定稳定存在的 stLayoutWrapper > stVerticalBlock ---- */
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo) {
        background: #ffffff !important;
        border-radius: 20px !important;
        padding: 40px !important;
        width: 100%;
        box-shadow: 0 15px 35px rgba(115, 174, 82, .10) !important;
        gap: 12px !important;
    }
    /* Tabs 下方 20px = gap 12 + 此处 8px */
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stElementContainer"]:has([data-testid*="segmented_control"]) {
        margin-bottom: 8px !important;
    }
    /* 复选框与按钮间距 20px = gap 12 + 此处 8px */
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stCheckbox"] { margin-bottom: 8px !important; }

    /* ---- 登录方式切换：自然紧凑、整组水平居中、按钮间 32px 间隙（对标参考图） ---- */
    [data-testid="stElementContainer"]:has([data-testid*="segmented_control"]) {
        align-self: center !important;
    }
    [data-testid="stButtonGroup"]:has([data-testid*="segmented_control"]),
    [data-testid="stButtonGroup"]:has([data-testid*="segmented_control"]) > div,
    [data-testid="stButtonGroup"]:has([data-testid*="segmented_control"]) > div > div {
        display: flex !important; gap: 16px !important; flex-wrap: nowrap !important;
    }
    [data-testid="stBaseButton-segmented_control"],
    [data-testid="stBaseButton-segmented_controlActive"] {
        flex: 0 0 auto !important; justify-content: center !important;
        font-size: .85rem !important; padding: 6px 16px !important;
        white-space: nowrap !important; border-radius: 8px !important;
    }
    [data-testid="stBaseButton-segmented_control"] {
        background: transparent !important; color: #7F8C8D !important;
        border: none !important; font-weight: 500;
    }
    [data-testid="stBaseButton-segmented_controlActive"] {
        background: #73AE52 !important; color: #fff !important;
        border: none !important; font-weight: 600;
    }

    /* ---- 登录方式切换（云端最新版 Streamlit 健壮兼容）----
       新版前端把 segmented_control 迁移为 BaseWeb TabList 渲染：
       外层 data-testid="stSegmentedControl"（不是 stTabs），内层 [data-baseweb="tab-list"]，
       按钮 [data-baseweb="tab"] + role="tab"。旧选择器（stButtonGroup / stBaseButton-*）
       在新 DOM 上全部落空导致左对齐，以下规则新旧两代 DOM 同时覆盖、强制居中均匀分布。 */
    [data-testid="stSegmentedControl"],
    div[data-testid="stTabs"] {
        width: 100% !important;
    }
    [data-testid="stSegmentedControl"] [data-baseweb="tab-list"],
    [data-testid="stSegmentedControl"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;   /* 强制整组居中 */
        align-items: center !important;
        gap: 20px !important;
        width: 100% !important;
        flex-wrap: nowrap !important;
        border-bottom: none !important;       /* BaseWeb TabList 自带下边框线，移除 */
    }
    /* 三个 Tab 平均分配宽度（flex:1）→ 均匀分布；去 flex:1 即紧凑居中 */
    [data-testid="stSegmentedControl"] button[data-baseweb="tab"],
    [data-testid="stSegmentedControl"] [role="tab"],
    [data-testid="stSegmentedControl"] [data-testid*="stBaseButton-segmented_control"],
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        flex: 1 1 0 !important;
        justify-content: center !important;
        text-align: center !important;
        font-size: .85rem !important;
        padding: 6px 16px !important;
        white-space: nowrap !important;
        border-radius: 8px !important;
    }
    /* 未选中态：透明底灰字（覆盖 BaseWeb 默认样式与 hover 底色） */
    [data-testid="stSegmentedControl"] button[data-baseweb="tab"],
    [data-testid="stSegmentedControl"] [role="tab"] {
        background: transparent !important;
        color: #7F8C8D !important;
        border: none !important;
        font-weight: 500;
    }
    [data-testid="stSegmentedControl"] button[data-baseweb="tab"]:hover,
    [data-testid="stSegmentedControl"] [role="tab"]:hover {
        background: rgba(115, 174, 82, .08) !important;
        color: #2C3E50 !important;
    }
    /* 选中态：绿色药丸（同时兼容 aria-selected 新结构与旧 Active 类名） */
    [data-testid="stSegmentedControl"] button[aria-selected="true"],
    [data-testid="stSegmentedControl"] [role="tab"][aria-selected="true"],
    [data-testid="stBaseButton-segmented_controlActive"] {
        background: #73AE52 !important;
        color: #fff !important;
        border: none !important;
        font-weight: 600;
    }

    /* ---- 输入框：高 46px、#F9FAFB 填充、无边框、圆角 8px ---- */
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stTextInput"] input {
        background: #F9FAFB !important; border: none !important;
        border-radius: 8px !important; height: 46px; padding: 0 14px !important;
        color: #2C3E50 !important;
    }
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stTextInput"] input:focus {
        background: #F3F6F0 !important;
        box-shadow: inset 0 -2px 0 #73AE52 !important;
        outline: none !important;
    }
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stTextInput"] label { font-size: .9rem; color: #2C3E50 !important; }

    /* ---- 复选框：小字灰色、两端对齐（整齐不歪斜） ---- */
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stCheckbox"] label {
        color: #7F8C8D !important; font-size: .82rem;
        text-align: justify; text-justify: inter-ideograph;
    }

    /* ---- 登录按钮：全宽 48px 苹果青，悬停轻微上浮 ---- */
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    button[data-testid="stBaseButton-primary"] {
        background-color: #73AE52 !important;
        border: 1px solid #73AE52 !important;
        color: #ffffff !important; height: 48px; font-size: 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(115, 174, 82, .20) !important;
        transition: all .18s ease;
    }
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #629A44 !important; border-color: #629A44 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(98, 154, 68, .32) !important;
    }

    /* ---- "阅读协议" expander（卡片内底部）：小字弱化样式 ---- */
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stExpander"] { border: none !important; background: transparent !important; }
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stExpander"] details { border: none !important; background: transparent !important; }
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stExpander"] summary {
        font-size: .8rem; color: #7F8C8D; min-height: 0; padding: 2px 0;
    }
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stExpander"] summary:hover { color: #629A44; }
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(#login-card-logo)
    [data-testid="stExpander"] [data-testid="stMarkdown"] { font-size: .84rem; color: #7F8C8D; }

    /* ---- 窄屏（≤900px）：自动折叠为单列（列选择器带伪类对齐特异性） ----
       列的垂直居中/最小高度由上方"右表单区"规则统一管理，此处只收窄宽度 */
    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"]:has(#brand-main) {
            flex-direction: column !important;
        }
        [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:first-child,
        [data-testid="stHorizontalBlock"]:has(#brand-main) > div[data-testid="stColumn"]:last-child {
            width: 100% !important;
        }
    }
</style>
"""

# 左侧品牌展示区：真实 Logo 图片（assets/logo.png 优先，兼容 logo设计.png）+ 文字组 +
# 底部隐私声明——全部合成【自绘 HTML 块】（图片 base64 内嵌），布局完全可控，
# 不受 Streamlit markdown DOM 处理（h1 id 重写/内联样式剥离）影响
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGO_CANDIDATES = [
    os.path.join(_APP_DIR, "assets", "logo.png"),
    os.path.join(_APP_DIR, "assets", "logo设计.png"),
]
# 取第一个实际存在的候选；都不存在时为 None → 左栏走纯文字兜底，页面不崩溃
_LOGO_PATH = next((p for p in _LOGO_CANDIDATES if os.path.exists(p)), None)


def _load_logo_b64():
    """读取 Logo 图片 → 缩至 560px（2x retina 足够）→ WebP 压缩 → base64 内嵌字符串"""
    if not _LOGO_PATH:
        return ""
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(_LOGO_PATH).convert("RGBA")
        img.thumbnail((560, 560))
        buf = BytesIO()
        img.save(buf, "WEBP", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        try:
            with open(_LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return ""


_LOGO_B64 = _load_logo_b64()

# 主内容块（图片 + 文字 + 三个功能卡片）与底部声明：**纯结构 HTML**（id/class 锚点），
# 布局样式全部放在 LOGIN_PAGE_CSS 的 #brand-main 选择器里——Streamlit 会剥离
# st.markdown HTML 的内联布局样式，关键布局绝不能依赖内联 style
BRAND_MAIN_HTML = f"""
<div id="brand-main">
    <img src="data:image/webp;base64,{_LOGO_B64}" alt="智学引擎">
    <div class="bp-title">智学引擎</div>
    <div class="bp-en">AI-POWERED LEARNING</div>
    <div class="bp-sub">让 AI 成为你的专属学习伴侣</div>
    <div class="bp-cards">
        <div class="bp-card">
            <div class="bp-card-title">知识图谱</div>
            <div class="bp-card-desc">自动构建课件知识点网络，先修依赖与核心概念一目了然。</div>
        </div>
        <div class="bp-card">
            <div class="bp-card-title">认知诊断</div>
            <div class="bp-card-desc">AI 按知识点精准出题，定位薄弱环节，量化学习增益。</div>
        </div>
        <div class="bp-card">
            <div class="bp-card-title">智能问答</div>
            <div class="bp-card-desc">基于你的课件内容作答，句句标注来源页码，可溯源不编造。</div>
        </div>
    </div>
</div>
"""

# 底部隐私声明：独立块，LOGIN_PAGE_CSS 将其推到左栏最底部并水平居中
BRAND_FOOT_HTML = """
<div id="brand-foot">数据物理隔离 · 不用于模型训练 · 不与其他用户共享</div>
"""


# 右侧卡片顶部：小号花形 Logo + "欢迎回来"（居中）
LOGIN_CARD_LOGO_HTML = """
<div id="login-card-logo" style="text-align:center; margin-bottom:2px;">
    <svg width="52" height="52" viewBox="0 0 72 72" aria-label="智学引擎"
         style="display:block; margin:0 auto 10px;">
        <g fill="#73AE52">
            <ellipse cx="36" cy="17.5" rx="9.5" ry="16" opacity=".92" transform="rotate(0 36 36)"/>
            <ellipse cx="36" cy="17.5" rx="9.5" ry="16" opacity=".78" transform="rotate(60 36 36)"/>
            <ellipse cx="36" cy="17.5" rx="9.5" ry="16" opacity=".64" transform="rotate(120 36 36)"/>
            <ellipse cx="36" cy="17.5" rx="9.5" ry="16" opacity=".92" transform="rotate(180 36 36)"/>
            <ellipse cx="36" cy="17.5" rx="9.5" ry="16" opacity=".78" transform="rotate(240 36 36)"/>
            <ellipse cx="36" cy="17.5" rx="9.5" ry="16" opacity=".64" transform="rotate(300 36 36)"/>
        </g>
        <circle cx="36" cy="36" r="7.5" fill="#ffffff"/>
        <circle cx="36" cy="36" r="4" fill="#629A44"/>
    </svg>
    <div style="font-size:1.3rem; font-weight:800; color:#2C3E50; line-height:1.25; letter-spacing:1px;">智学引擎</div>
    <div style="font-size:.92rem; color:#7F8C8D; margin-top:4px;">欢迎回来</div>
</div>
"""


# 协议摘要弹层内容（与《用户协议》《隐私政策》帮助中心全文一致的核心条款）
_TOS_SUMMARY = (
    "**隐私保护**：本系统尊重用户隐私，您上传的文档仅保存在您的独立存储空间，"
    "不会用于训练大模型，也不会被其他用户共享。\n\n"
    "**内容规范**：请确保您上传的文档拥有合法版权或使用权，"
    "切勿上传涉密、侵权或违规内容。"
)


def page_login():
    """
    登录页（左右分栏 SaaS 风格）：
      - 左半：品牌展示区（渐变深色背景 + 特色功能点，见 BRAND_PANE_HTML）；
      - 右半：白色登录卡片，含 登录 / 注册 / 管理员 三种身份入口（segmented 切换）。
    实现要点：
      - 不用 st.form：表单内 widget 交互不触发重跑，协议复选框的 disabled 状态无法
        实时联动按钮；改用裸控件 + 按钮 disabled，勾选/取消即时生效。
      - 不用 st.tabs：tabs 无法放进卡片容器，改用 st.segmented_control 切换三种入口。
      - 复选框紧贴登录按钮（用户要求），未勾选时按钮不可点击。
    登录成功后 user_name / role 写入 st.session_state，刷新后进入对应端。
    """
    st.markdown(LOGIN_PAGE_CSS, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="small")   # 左品牌 50% / 右表单 50%
    with left:
        # 主内容块（图 + 文，80vh 内双居中）+ 底部声明（CSS 推到最底部）
        st.markdown(BRAND_MAIN_HTML, unsafe_allow_html=True)
        st.markdown(BRAND_FOOT_HTML, unsafe_allow_html=True)

    with right:
        # 右：登录卡片（st.container(border=True) 生成真实容器 div，CSS 美化成白卡）
        with st.container(border=True):
            st.markdown(LOGIN_CARD_LOGO_HTML, unsafe_allow_html=True)

            # ---- 登录方式切换（默认"登录"；label 隐藏但保留可访问性语义） ----
            tab = st.segmented_control(
                "登录方式", ["登录", "注册", "管理员"],
                default="登录", key="login_tab", label_visibility="collapsed",
            ) or "登录"

            # ---- 登录：昵称 + 密码（SQLite 加盐哈希校验） ----
            if tab == "登录":
                name = st.text_input("昵称", key="login_name", max_chars=20,
                                     placeholder="你注册时使用的昵称")
                pwd = st.text_input("密码", type="password", max_chars=64, key="login_pwd")
                # 合规前置：未勾选协议 -> 按钮禁用（灰态不可点击）
                agree = st.checkbox("我已阅读并同意《用户协议》和《隐私政策》",
                                    key="agree_tos")
                if st.button("登录", type="primary", use_container_width=True,
                             disabled=not agree):
                    try:
                        login(name, pwd)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))   # 昵称为空 / 未注册 / 密码错误 / 旧账号未认领

            # ---- 注册：昵称 + 密码 + 确认密码（成功即登录） ----
            elif tab == "注册":
                r_name = st.text_input("昵称", key="reg_name", max_chars=20,
                                       placeholder="给自己起一个好记的昵称（唯一，≤20字）")
                r_pwd = st.text_input("密码（至少 4 位）", type="password",
                                      max_chars=64, key="reg_pwd")
                r_pwd2 = st.text_input("确认密码", type="password",
                                       max_chars=64, key="reg_pwd2")
                # 合规前置：注册即视为同意协议，未勾选 -> 按钮禁用
                agree = st.checkbox("我已阅读并同意《用户协议》和《隐私政策》",
                                    key="agree_tos")
                if st.button("注册并进入学习", type="primary",
                             use_container_width=True, disabled=not agree):
                    if r_pwd != r_pwd2:
                        st.error("两次输入的密码不一致")
                    else:
                        try:
                            register(r_name, r_pwd)
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))   # 昵称为空 / 已被注册 / 密码过短

            # ---- 管理员入口：昵称 + 管理密码（.env 的 ADMIN_PASSWORD），role=admin ----
            else:
                admin_name = st.text_input("管理员昵称", key="login_a_name", max_chars=20,
                                           placeholder="开发者昵称")
                pwd = st.text_input("管理密码", type="password", max_chars=64, key="login_a_pwd")
                # 管理员为系统所有者，协议复选框不适用
                if st.button("进入开发者端", type="primary", use_container_width=True):
                    try:
                        login_admin(admin_name, pwd)
                        st.rerun()
                    except (ValueError, PermissionError) as e:
                        st.error(str(e))   # 昵称为空 / 密码错误 / .env 未配置

            # ---- "阅读协议" expander（卡片内底部，小字弱化样式见 CSS） ----
            with st.expander("阅读《用户协议》及《隐私政策》"):
                st.markdown(_TOS_SUMMARY)


def do_logout():
    """
    退出登录：清空全部会话状态，回到登录页。
    实现：auth 层记录日志并复位身份 -> 设置 logout_pending 标志 -> 触发重跑。
    键删除不在本函数执行——此刻 radio（key="page"）已实例化，
    在脚本运行中删除 widget key 会抛 StreamlitAPIException 导致清理失败
    （表现为"退出后仍是登录态 / 换角色登录后页面崩溃"）；
    真正的清理在 main() 开头、widget 实例化之前完成。
    """
    logout()   # 记录登出日志、复位身份（要在清空前拿到用户信息）
    st.session_state["logout_pending"] = True
    st.rerun()


# ========== 开发者端页面（仅 admin 角色可见） ==========
ROLE_BADGE = {"student": "学生", "admin": "开发者"}


def page_admin_overview():
    """开发者端：全平台数据总览（用户 / 学习行为 / API 消耗 / 近 7 天活跃度）"""
    from config import APP_VERSION
    st.title("数据总览")
    st.caption(f"全平台运行数据一览（管理员视角，不受个人身份过滤） · 版本 v{APP_VERSION}")
    data = get_admin_overview()
    # ---- 核心指标：用户 / 文件 / 问答 / 诊断 ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总用户数", f"{data['users']:,}")
    c2.metric("总文件数", f"{data['files']:,}")
    c3.metric("总问答次数", f"{data['qa']:,}")
    c4.metric("总诊断次数", f"{data['diagnosis']:,}")
    # ---- 次级指标：路径 / 评估 / API ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("路径生成", f"{data['path']:,}")
    c2.metric("评估次数", f"{data['eval']:,}")
    c3.metric("API 调用", f"{data['requests']:,}")
    c4.metric("总消费", f"¥{data['total_cost']:.4f}")
    # ---- 第三行：Tokens 总消耗 + data/ 专属存储区磁盘总占用 ----
    disk_mb = storage.total_size_bytes() / (1024 * 1024)
    n_user_dirs = (len([d for d in storage.BASE_DIR.iterdir() if d.is_dir()])
                   if storage.BASE_DIR.is_dir() else 0)
    c1, c2, c3, _ = st.columns(4)
    c1.metric("总 Tokens", f"{data['total_tokens']:,}")
    c2.metric("磁盘总占用", f"{disk_mb:.2f} MB",
              help="data/ 目录下全部用户专属文件夹的课件原件占用（多租户物理存储）")
    st.caption(f"课件原件物理存储：`data/`（按用户分目录隔离，共 {n_user_dirs} 个用户目录）")
    st.divider()

    # ---- 近 7 天全平台活跃度折线图（上传/问答/诊断/路径/评估/API调用 按天计数） ----
    st.subheader("近 7 天活跃度")
    activity = get_daily_activity(days=7)
    if activity:
        st.line_chart(activity, height=260)
        st.caption("纵轴：当日事件数（上传、问答、诊断、路径、评估、API 调用之和）；横轴：日期")
    else:
        st.info("近 7 天暂无活动记录。")
    if st.button("刷新数据"):
        st.rerun()


def page_admin_users():
    """开发者端：用户管理（全用户列表 + 点击查看单个用户的详细记录）"""
    import pandas as pd

    st.title("用户管理")
    st.caption("全部注册用户的角色、上传量、消费与最后活跃时间；选中用户可查看详细记录")

    stats = get_user_stats()
    if not stats:
        st.info("暂无注册用户。")
        return

    # ---- 全用户列表：昵称 / 角色 / 状态 / 上传文件数 / 消费金额 / 最后活跃时间 ----
    rows = [{
        "昵称": u["user_name"],
        "角色": ROLE_BADGE.get(u["role"], u["role"]),
        "状态": "已禁用" if u.get("is_disabled") else "正常",
        "上传文件数": u["files"],
        "消费金额(元)": round(u["cost"], 4),
        "最后活跃时间": u["last_active"] or "-",
        "注册时间": u["created_at"],
    } for u in stats]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- 点击某用户查看详细记录：学习足迹 + 最近 API 调用 ----
    st.divider()
    name = st.selectbox("选择用户查看详细记录",
                        [u["user_name"] for u in stats],
                        index=None, placeholder="点击选择一个用户…")
    if not name:
        return
    user = next(u for u in stats if u["user_name"] == name)
    st.markdown(f"#### {ROLE_BADGE.get(user['role'], user['role'])} {name} 的详细记录")
    b1, b2, b3 = st.columns(3)
    b1.metric("上传文件数", user["files"])
    b2.metric("消费金额", f"¥{user['cost']:.4f}")
    b3.metric("最后活跃", (user["last_active"] or "-")[:16])

    st.markdown("**学习记录**")
    records = get_history(name, user["role"])
    if records:
        rec_rows = [{"类型": TYPE_LABELS.get(r["type"], r["type"]),
                     "文件": r["filename"] or "-", "时间": r["time"]} for r in records[:30]]
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("暂无学习记录")

    st.markdown("**最近 API 调用（最多 20 条）**")
    calls = [c for c in get_recent_api_usage(100) if c["user_name"] == name][:20]
    if calls:
        call_rows = [{"时间": c["timestamp"], "输入 tokens": c["prompt_tokens"],
                      "输出 tokens": c["completion_tokens"],
                      "费用(元)": round(c["estimated_cost"], 6)} for c in calls]
        st.dataframe(pd.DataFrame(call_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("暂无 API 调用记录")

    # ---- 课件文件列表（只读）：该用户专属目录 data/{name}/ 的原件与数据库记录 ----
    st.divider()
    st.markdown("**课件文件列表（只读）**")
    st.caption("来自 files 表（按用户过滤）与 data/ 专属目录的物理原件；管理员仅可查看，不可操作")
    user_files = list_files(user_name=name, role=user["role"])
    if user_files:
        file_rows = [{
            "文件名": f["filename"],
            "上传时间": f["upload_time"],
            "关联记录": f["qa"] + f["diagnosis"] + f["path"] + f["eval"],
            "物理路径": f.get("store_path") or "（旧数据，未落盘）",
        } for f in user_files]
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("该用户尚未上传任何课件")

    # ---- 账号管控：禁用 / 启用（admin 账号不可禁用，防锁死） ----
    st.divider()
    st.markdown("**账号管控**")
    if user["role"] == "admin":
        st.caption("admin 账号不允许禁用（防止把管理员锁在门外）。")
    elif user.get("is_disabled"):
        st.warning(f"账号「{name}」当前处于 **禁用** 状态：无法登录，登录中的会话会被强制下线。")
        if st.button(f"启用账号「{name}」", type="primary"):
            if set_user_disabled(name, False):
                log_event("user_enable", f"启用账号「{name}」",
                          user_name=st.session_state.user_name, role="admin")
                st.toast(f"已启用账号「{name}」。")
                st.rerun()
            else:
                st.error("操作失败，请重试。")
    else:
        st.caption(f"账号「{name}」当前 **正常**。禁用后该用户将无法登录，已有会话在下次交互时被踢出。")
        confirm_key = f"confirm_disable_{name}"
        if st.button(f"禁用账号「{name}」"):
            st.session_state[confirm_key] = True
        # 二次确认（防误操作）：勾选后才真正写入禁用标记
        if st.session_state.get(confirm_key):
            if st.checkbox(f"确认禁用「{name}」？该用户将立即无法登录。"):
                if set_user_disabled(name, True):
                    log_event("user_disable", f"禁用账号「{name}」",
                              user_name=st.session_state.user_name, role="admin")
                    st.session_state[confirm_key] = False
                    st.toast(f"已禁用账号「{name}」。")
                    st.rerun()
                else:
                    st.error("操作失败，请重试。")


def page_admin_usage():
    """开发者端：全平台用量（DeepSeek 平台"用量信息"后台式布局，数据全部真实读库）：
      顶部两大核心指标卡（充值余额 / 累计消费金额）→ 中间三张关键指标卡
      （消费金额 / API请求次数 / Tokens 总量）→ 底部近 30 天每日消费折线图
      + 用户消费排行榜 Top 10。
    访问控制：本页注册于 ADMIN_PAGES，学生端访问会被 main() 导航校验拦截并跳回首页。
    """
    import pandas as pd

    st.title("全平台用量")
    st.caption("所有用户的 API 消耗统计（计费口径：输入 1 元/百万 tokens、输出 2 元/百万 tokens）")
    summary = get_api_usage_summary()   # user_name/role 均为 None = 全平台聚合

    # ---- 1. 顶部：两大核心指标卡 ----
    # 充值余额：从 .env 的 DEEPSEEK_BALANCE 读取（平台不提供余额查询 API，充值后手动更新）
    try:
        balance = float(os.getenv("DEEPSEEK_BALANCE", "0"))
    except ValueError:
        balance = 0.0
    top_left, top_right = st.columns(2)
    top_left.metric("充值余额", f"¥{balance:.2f}",
                    help="来自 .env 的 DEEPSEEK_BALANCE，请在充值后手动更新该值")
    top_right.metric("累计消费金额", f"¥{summary['total_cost']:.4f}",
                     help="api_usage 表全部记录费用之和（本地估算口径）")

    st.divider()

    # ---- 2. 中间：三张关键指标卡（全部真实聚合自 api_usage 表） ----
    mid_cost, mid_req, mid_tok = st.columns(3)
    mid_cost.metric("消费金额", f"¥{summary['total_cost']:.4f}")
    mid_req.metric("API请求次数", f"{summary['requests']:,}")
    mid_tok.metric("Tokens 总量", f"{summary['total_tokens']:,}")

    st.divider()

    # ---- 3. 底部：趋势图表 ----
    # 近 30 天每日消费折线图：api_usage 按日期分组（YYYY-MM-DD 字典序=时间序，直接比较）
    st.subheader("每日消费趋势（近 30 天）")
    daily = get_api_usage_daily()
    cutoff = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    daily_30 = {d: c for d, c in daily.items() if d >= cutoff}
    if daily_30:
        st.line_chart(daily_30, height=280)
        peak_day = max(daily_30, key=daily_30.get)
        st.caption(f"纵轴：当日消费（元）。近 30 天峰值出现在 {peak_day}"
                   f"（¥{daily_30[peak_day]:.4f}）")
    else:
        st.info("近 30 天暂无用量记录。")

    # 用户消费排行榜 Top 10（按消费降序，真实聚合自 api_usage 表）
    st.subheader("用户消费排行榜 Top 10")
    by_user = get_api_usage_by_user()[:10]
    if by_user:
        rows = [{"排名": i + 1, "用户名": u["user_name"],
                 "角色": ROLE_BADGE.get(u["role"], u["role"]),
                 "API请求次数": u["requests"], "Tokens": u["tokens"],
                 "消费金额(元)": round(u["cost"], 4)}
                for i, u in enumerate(by_user)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("暂无消费记录。")

    st.divider()
    # ---- 充值入口：跳转 DeepSeek 控制台 ----
    st.markdown("[前往 DeepSeek 控制台充值](https://platform.deepseek.com/usage)")


def page_admin_monitor():
    """
    开发者端：系统监控（运营健康度一览）：
      - 存活体检：数据库 / DeepSeek API / 磁盘 / 错误日志四项真实探测；
      - 每日 API 消耗趋势（api_usage 表按天汇总，全平台）；
      - 每日新增注册用户数（users 表按 created_at 按天计数）；
      - 折线图共用同一时间轴，便于对照"增长 -> 消耗"的关联关系。
    """
    st.title("系统监控")
    st.caption("全平台运营健康度：存活体检 + API 消耗与用户增长趋势（管理员视角，不受个人身份过滤）")

    # ---- 存活体检（真实探测；结果缓存 session_state，点「重新体检」才重新探测） ----
    # 缓存原因：API 探测是真实请求（会消耗 token），不能每次 rerun 都发一遍。
    st.subheader("存活体检")
    if "health_check" not in st.session_state:
        with st.spinner("首次体检中：正在探测数据库 / DeepSeek API / 磁盘 / 错误日志…"):
            st.session_state["health_check"] = run_health_check()
    hc = st.session_state["health_check"]

    def _pill(ok):
        """状态圆点（几何符号 + Streamlit 彩色文本，非 emoji）：正常绿 / 异常红"""
        return ":green[●]" if ok else ":red[●]"

    col_db, col_api, col_disk, col_log = st.columns(4)
    with col_db:
        ok, detail = hc["db"]
        st.markdown(f"{_pill(ok)} **数据库连通性**")
        st.caption(("" if ok else "") + detail)
    with col_api:
        ok, detail = hc["api"]
        st.markdown(f"{_pill(ok)} **DeepSeek API**")
        st.caption(("" if ok else "") + detail)
    with col_disk:
        st.markdown("**磁盘空间**")
        disk_lines = []
        for label, used_pct, free_gb in hc["disk"]:
            if used_pct is None:   # 盘不存在（如 Linux 容器里没有 C:）
                disk_lines.append(f"{label}：不可探测")
            else:
                disk_lines.append(f"{label}：剩 {free_gb:.1f} GB（已用 {used_pct:.0f}%）")
        st.caption("\n\n".join(disk_lines))
    with col_log:
        y_err = hc["errors"]["yesterday"]
        t_err = hc["errors"]["today"]
        ok = (y_err == 0 and t_err == 0)
        st.markdown(f"{_pill(ok)} **错误日志**")
        st.caption(f"昨天 {y_err} 条 · 今天 {t_err} 条"
                   + ("，请到「系统日志」排查" if not ok else "，一切正常"))

    b_hc, _ = st.columns([1, 3])
    with b_hc:
        if st.button("重新体检", type="primary"):
            with st.spinner("体检中：正在探测数据库 / DeepSeek API / 磁盘 / 错误日志…"):
                st.session_state["health_check"] = run_health_check()
            st.rerun()
    st.caption(f"上次体检时间：{hc['checked_at']}（API 探测为真实 1-token 请求，缓存避免重复消耗）")

    st.divider()

    # ---- 每日 API 消耗折线图 ----
    st.subheader("每日 API 消耗")
    daily_cost = get_api_usage_daily()
    if daily_cost:
        st.line_chart(daily_cost, height=260)
        peak_day = max(daily_cost, key=daily_cost.get)
        st.caption(f"纵轴：当日估算消费（元）；横轴：日期。峰值出现在 {peak_day}"
                   f"（¥{daily_cost[peak_day]:.4f}）")
    else:
        st.info("暂无 API 用量记录。")

    st.divider()

    # ---- 每日新增用户折线图 ----
    st.subheader("每日新增用户")
    reg_daily = get_registration_daily()
    if reg_daily:
        st.line_chart(reg_daily, height=260)
        st.caption(f"纵轴：当日注册用户数；横轴：日期。累计注册 {sum(reg_daily.values())} 人。")
    else:
        st.info("暂无注册记录。")

    st.divider()
    if st.button("刷新监控数据"):
        st.rerun()


def page_admin_logs():
    """开发者端：系统日志（API 调用明细 + 错误日志）"""
    import pandas as pd

    st.title("系统日志")
    tab_api, tab_err = st.tabs(["API 调用记录", "错误日志"])

    # ---- 最近 100 条 API 调用明细 ----
    with tab_api:
        calls = get_recent_api_usage(100)
        if not calls:
            st.info("暂无 API 调用记录。")
        else:
            rows = [{
                "时间": c["timestamp"], "用户": c["user_name"],
                "角色": ROLE_BADGE.get(c["role"], c["role"] or "-"),
                "输入 tokens": c["prompt_tokens"], "输出 tokens": c["completion_tokens"],
                "总 tokens": c["total_tokens"], "费用(元)": round(c["estimated_cost"], 6),
            } for c in calls]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("展示最近 100 条调用；费用为本地估算（输入 1 元/百万 + 输出 2 元/百万 tokens）")

    # ---- 错误日志：API 调用失败时由 BaseAgent 持久化到 event_log（event="api_error"） ----
    with tab_err:
        st.caption(
            "API 调用失败（余额/超时/网络/未知异常）会被自动记录到数据库；"
            "agent 实例的 last_error 仅为本会话内存态，跨会话审计以此表为准。"
        )
        logs = get_recent_logs(100, event="api_error")
        if not logs:
            st.success("暂无错误记录，服务运行正常。")
            return
        rows = [{
            "时间": r["timestamp"], "用户": r["user_name"],
            "错误": r["detail"] or "",
        } for r in logs]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ========== 管理员页 5：Agent 配置 ==========
# 开关清单：(配置键, 展示名, 说明)——每项一个 st.toggle，停用后学习端对应入口给出统一提示
AGENT_SWITCHES = [
    ("parser_enabled", "解析 Agent",
     "上传课件的文本切块与 AI 知识点抽取。停用后学习端点击「开始解析」会收到停用提示。"),
    ("planner_enabled", "自主规划 Agent",
     "AI 自主规划：无需上传课件，输入主题直接生成知识图谱与学习路径。停用后学习端「AI 自主规划」页的规划按钮不可用。"),
    ("ontology_enabled", "图谱 Agent",
     "AI 构建知识图谱：解析流水线第 4 步、图谱页重建、增量更新与诊断/评估的自动补图。停用后相关入口提示不可用。"),
    ("tutor_enabled", "问答 Tutor Agent",
     "智能问答的引导式回答（先引导思考、标注来源与知识点）。停用后学习端提问与「重新生成」都会收到停用提示。"),
    ("diagnosis_enabled", "诊断 Agent",
     "AI 出题与诊断报告生成。停用后学习端无法生成诊断题目或提交作答。"),
    ("path_enabled", "路径 Agent",
     "学习路径生成（本地拓扑排序，无 API 消耗）。停用后学习端无法生成路径。"),
    ("eval_enabled", "评估 Agent",
     "前后测出题与 AI 学习增益报告（快速评估也一并停用）。停用后学习端无法进行学习评估。"),
]


def page_admin_agents():
    """
    开发者端「Agent 配置」页：Agent 启停开关 + 执行参数。
    配置存入 st.session_state.agent_config（仅当前会话，不落库），
    学习端各功能在执行瞬间通过 get_agent_config() 实时读取——
    管理员改完配置，学习端下一次点击立即生效，无需重启。
    """
    st.title("Agent 配置")
    st.caption(
        "各 AI Agent 的启停开关与执行参数。配置保存在当前会话（st.session_state），"
        "修改即时生效：学习端执行相应功能时会实时读取这里的开关与参数。"
    )
    cfg = get_agent_config()

    # ---- 开关区：每个 Agent 一个 toggle，变化后立即同步回 agent_config ----
    st.subheader("Agent 开关")
    for key, name, desc in AGENT_SWITCHES:
        on = st.toggle(name, value=cfg[key], key=f"cfg_w_{key}", help=desc)
        if on != cfg[key]:   # toggle 拨动触发重跑，这里把新值写回配置（执行端统一读 agent_config）
            cfg[key] = on

    st.divider()
    # ---- 参数区：诊断出题数量（学习端滑块默认值）+ 检索置信度阈值（问答引用过滤） ----
    st.subheader("执行参数")
    c1, c2 = st.columns(2)
    n_q = c1.select_slider(
        "诊断出题数量", options=[3, 5, 10], value=cfg["diagnosis_n_questions"],
        key="cfg_w_nq",
        help="学习端「学习诊断」页出题数量滑块的默认值，生成题目时读取",
    )
    cfg["diagnosis_n_questions"] = n_q
    thr = c2.slider(
        "检索置信度阈值", min_value=0.0, max_value=1.0, step=0.05,
        value=float(cfg["retrieval_min_score"]), key="cfg_w_thr",
        help="智能问答引用课件片段的最低混合相关度（0.7×向量相似度 + 0.3×图谱匹配）。"
             "低于阈值的片段不作为回答依据；调高可过滤不相关引用，过高可能导致无可引用片段。",
    )
    cfg["retrieval_min_score"] = round(float(thr), 2)

    st.divider()
    # ---- 恢复默认 + 当前配置预览 ----
    if st.button("恢复全部默认配置"):
        for k in list(st.session_state.keys()):
            if k.startswith("cfg_w_"):
                del st.session_state[k]   # 清掉 widget 键，下次渲染才会使用默认值
        st.session_state.agent_config = dict(DEFAULT_AGENT_CONFIG)
        st.rerun()
    with st.expander("查看当前配置（st.session_state.agent_config）"):
        st.json(get_agent_config())


# ========== 开发者端：任务监控与系统告警 ==========
TASK_TYPE_LABELS = {
    "parse_pdf": "解析 PDF",
    "topic_plan": "AI 自主规划",
    "diagnosis_quiz": "生成诊断题",
    "diagnosis_report": "诊断报告",
    "path_generate": "生成路径",
}


def _read_error_log_tail(lines=30):
    """读取 logs/error.log 最后 N 行（任务失败详情页关联展示）；文件不存在返回提示文本"""
    log_path = Path(__file__).resolve().parent / "logs" / "error.log"
    if not log_path.exists():
        return "（logs/error.log 尚未生成——还没有记录过任何错误）"
    text = log_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return "（logs/error.log 为空）"
    return "\n".join(text.splitlines()[-lines:])


def page_admin_tasks():
    """开发者端：任务监控（后台任务状态总览 + 近 7 天趋势 + 失败任务明细与 error.log 关联）"""
    import pandas as pd

    st.title("任务监控")
    st.caption("后台任务（解析 PDF / 生成诊断 / 生成路径）的执行状态、趋势与失败原因追溯")

    # ---- 状态总览：四态计数指标卡 ----
    counts = get_task_counts_by_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("待执行", counts["pending"])
    c2.metric("执行中", counts["processing"])
    c3.metric("成功", counts["success"])
    c4.metric("失败", counts["failed"], delta=None if counts["failed"] == 0 else 0,
              delta_color="inverse")

    st.divider()

    # ---- 近 7 天任务成功/失败趋势（st.line_chart，需求 3） ----
    st.subheader("近 7 天任务趋势")
    stats = get_task_daily_stats(days=7)
    if stats:
        df = pd.DataFrame(stats).set_index("day")[["success", "failed"]]
        st.line_chart(df, height=280)
        st.caption("纵轴：当日任务数；绿线=成功、红线=失败（悬停查看具体数值）。")
    else:
        st.info("近 7 天暂无任务记录——任务（解析/诊断/路径）执行后会出现在这里。")

    st.divider()

    # ---- 失败任务列表：点击选择后查看具体 error_msg，并关联 error.log 尾部 ----
    st.subheader("失败任务明细")
    failed = get_failed_tasks(limit=50)
    if not failed:
        st.success("暂无失败任务，一切运行正常。")
        return
    df_failed = pd.DataFrame([{
        "任务ID": f"#{t['id']}",
        "用户": t["user_name"] or "-",
        "类型": TASK_TYPE_LABELS.get(t["task_type"], t["task_type"]),
        "失败时间": t["updated_at"],
        "失败原因": (t["error_msg"] or "（无错误信息）")[:60],
    } for t in failed])
    st.dataframe(df_failed, use_container_width=True, hide_index=True)

    picked = st.selectbox(
        "选择失败任务查看详情（含 error.log 关联日志）",
        [f"#{t['id']} · {TASK_TYPE_LABELS.get(t['task_type'], t['task_type'])} · {t['user_name']}"
         for t in failed],
        index=None, placeholder="点击选择一个失败任务…",
    )
    if picked:
        idx = [f"#{t['id']} · {TASK_TYPE_LABELS.get(t['task_type'], t['task_type'])} · {t['user_name']}"
               for t in failed].index(picked)
        t = failed[idx]
        st.error(f"任务 #{t['id']}「{TASK_TYPE_LABELS.get(t['task_type'], t['task_type'])}」"
                 f"失败原因：{t['error_msg'] or '（无错误信息）'}")
        st.caption(f"发生时间：{t['created_at']} → {t['updated_at']}")
        with st.expander("关联日志：logs/error.log 最后 30 行", expanded=False):
            st.code(_read_error_log_tail(30), language=None)
    if st.button("刷新任务数据"):
        st.rerun()


def page_admin_alerts():
    """开发者端：系统告警（磁盘空间 / 失败任务激增 / 卡死任务 / 最近错误日志）"""
    import shutil
    from pathlib import Path as _Path

    st.title("系统告警")
    st.caption("磁盘空间、任务失败激增与卡死任务的自动巡检（每次进入本页即重新检查）")

    # ---- 告警 1：本地磁盘空间（shutil.disk_usage，需求 3） ----
    st.subheader("本地磁盘空间")
    usage = shutil.disk_usage(_Path(__file__).resolve().anchor)   # 项目所在盘符
    total_gb = usage.total / 1024 ** 3
    free_gb = usage.free / 1024 ** 3
    used_pct = usage.used / usage.total if usage.total else 0
    st.progress(min(used_pct, 1.0), text=f"磁盘已使用 {used_pct:.0%}")
    d1, d2, d3 = st.columns(3)
    d1.metric("总容量", f"{total_gb:.1f} GB")
    d2.metric("已使用", f"{(usage.used / 1024 ** 3):.1f} GB")
    d3.metric("剩余可用", f"{free_gb:.1f} GB")
    if free_gb < 1:
        st.error("告警：磁盘剩余空间不足 1GB！继续写入课件与日志可能失败，请立即清理磁盘。")
    elif free_gb < 5:
        st.warning(f"磁盘剩余空间仅 {free_gb:.1f} GB，建议尽快清理。")
    else:
        st.success("磁盘空间充足。")

    st.divider()

    # ---- 告警 2：失败任务激增（近 24 小时失败次数） ----
    st.subheader("任务失败巡检")
    c24 = get_task_counts_since(hours=24)
    if c24["total"] == 0:
        st.info("近 24 小时暂无任务。")
    elif c24["failed"] >= 10:
        st.error(f"告警：近 24 小时失败任务 {c24['failed']} / {c24['total']}，"
                 "失败率异常偏高，请前往「任务监控」查看失败原因！")
    elif c24["failed"] >= 3:
        st.warning(f"近 24 小时失败任务 {c24['failed']} / {c24['total']}，建议关注。")
    else:
        st.success(f"近 24 小时任务 {c24['total']} 个，失败 {c24['failed']} 个，运行正常。")

    st.divider()

    # ---- 告警 3：卡死任务（processing 超过 15 分钟无状态更新） ----
    st.subheader("卡死任务巡检")
    stuck = get_stuck_tasks(minutes=15)
    if stuck:
        rows = "\n".join(f"- 任务 #{t['id']}「{TASK_TYPE_LABELS.get(t['task_type'], t['task_type'])}」"
                         f"（用户：{t['user_name'] or '-'}，自 {t['updated_at']} 起无更新）" for t in stuck)
        st.warning(f"以下任务长时间停留在「执行中」，疑似执行线程异常退出：\n\n{rows}")
    else:
        st.success("没有卡死任务。")

    st.divider()

    # ---- 告警 4：最近错误日志（error.log 尾部，需求 3 的"点击查看 error_msg"关联入口） ----
    st.subheader("最近错误日志")
    st.caption("来自 logs/error.log 的最后 20 行（完整日志请到服务器查看）")
    with st.expander("展开查看", expanded=False):
        st.code(_read_error_log_tail(20), language=None)
    if st.button("重新巡检"):
        st.rerun()


# ========== 开发者端：系统维护（存储管理 + 孤儿/旧日志清理） ==========
def page_admin_maintenance():
    """开发者端：系统维护（data/ 存储占用一览 + 孤儿文件与旧日志的真实清理）"""
    import pandas as pd

    st.title("系统维护")
    st.caption("存储占用一览与无用文件回收：孤儿课件原件、上传中断残留（.part）、7 天前的旧日志")

    # ---- 存储占用一览（需求 2）：data/ 总占用 + 各用户明细 ----
    st.subheader("课件存储占用（data/）")
    total_bytes = storage.total_size_bytes()
    st.metric("data/ 总占用", f"{total_bytes / 1048576:.2f} MB",
              help="全部用户专属目录的课件原件占用（含少量上传中断残留）")
    sizes = storage.user_dir_sizes()
    if sizes:
        df = pd.DataFrame([{"用户": u, "占用 (MB)": round(b / 1048576, 2), "文件数": n}
                           for u, b, n in sizes])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart({u: round(b / 1048576, 2) for u, b, _ in sizes}, height=240)
        st.caption("纵轴：该用户目录占用（MB）")
    else:
        st.info("暂无用户上传任何课件，data/ 目录为空。")

    st.divider()

    # ---- 清理盘点（只读预览，不删除） ----
    st.subheader("可清理项盘点")
    referenced = get_referenced_store_paths()   # 数据库仍引用的物理路径（清理白名单）
    orphans = storage.scan_orphan_files(referenced)
    old_logs = storage.list_old_logs(days=7)
    orphan_mb = sum(o[3] for o in orphans) / 1048576
    log_mb = sum(l[1] for l in old_logs) / 1048576

    if orphans:
        st.warning(f"发现 **{len(orphans)} 个孤儿文件**（约 {orphan_mb:.2f} MB）："
                   "磁盘上存在、但数据库已无任何记录引用（含上传中断残留的 .part 文件）。")
        with st.expander("查看孤儿文件清单"):
            st.dataframe(pd.DataFrame([{"用户": o[0], "文件": o[1], "大小 (KB)": round(o[3] / 1024, 1)}
                                       for o in orphans]), use_container_width=True, hide_index=True)
    else:
        st.success("没有孤儿文件——磁盘上的课件原件全部与数据库记录一一对应。")

    if old_logs:
        st.warning(f"发现 **{len(old_logs)} 个超过 7 天的旧日志**（约 {log_mb:.2f} MB）。")
    else:
        st.caption("日志文件均在 7 天保留期内，无需清理。")

    # ---- 立即清理（需求 3）：执行真实清理并汇报释放量 ----
    st.divider()
    st.subheader("执行清理")
    if st.button("立即清理", type="primary",
                 disabled=not (orphans or old_logs),
                 help="删除全部孤儿文件与超过 7 天的旧日志；被数据库引用的课件原件绝不触碰"):
        o_count, o_freed, paths = storage.cleanup_orphan_files(referenced)
        l_count, l_freed = storage.cleanup_old_logs(days=7)
        freed_mb = (o_freed + l_freed) / 1048576
        log_event("maintenance_cleanup",
                  f"清理孤儿文件 {o_count} 个、旧日志 {l_count} 个，释放 {freed_mb:.2f} MB",
                  user_name=st.session_state.user_name, role="admin")
        st.success(f"清理了 {o_count + l_count} 个无用文件，释放了 {freed_mb:.2f} MB 空间。")
        if paths:
            with st.expander("清理明细"):
                st.code("\n".join(paths), language=None)
    elif not (orphans or old_logs):
        st.caption("当前没有可清理的文件。")

    # ---- 数据库备份与恢复：真实落地 backup/ 时间戳备份 + 一键回滚 ----
    st.divider()
    st.subheader("数据库备份与恢复")
    backups = list_db_backups()
    c_bk, c_rs = st.columns(2)

    with c_bk:
        st.markdown("**立即备份**")
        st.caption("在线备份 learning_records.db 到 backup/ 目录（时间戳命名，原子一致）")
        if st.button("立即备份", use_container_width=True):
            name = backup_db()
            log_event("db_backup", f"数据库已备份 -> backup/{name}",
                      user_name=st.session_state.user_name, role="admin")
            st.success(f"已备份：backup/{name}")

    with c_rs:
        st.markdown("**查看备份**")
        if backups:
            sel = st.selectbox("选择一个备份文件（新 → 旧）", backups,
                               label_visibility="collapsed")
            st.caption("恢复会覆盖当前数据库；恢复前会自动备份当前数据作为保底快照。")
            if st.button("恢复此备份", use_container_width=True, type="primary"):
                pre = restore_db(sel)
                log_event("db_restore", f"数据库已从 backup/{sel} 恢复"
                          f"（恢复前快照 backup/{pre}）",
                          user_name=st.session_state.user_name, role="admin")
                st.success("恢复完成，页面即将刷新…")
                time.sleep(1)
                st.rerun()
        else:
            st.caption("尚无备份——点击左侧「立即备份」生成第一份。")


# ========== 学生端：帮助中心（产品使用说明） ==========
def page_help():
    """帮助中心：快速入门六步曲（带流程插图卡片）+ 常见问题 FAQ 折叠面板"""
    st.title("帮助中心")
    st.caption("三分钟上手智学 AI 助手——快速入门与常见问题都在这里")

    # ---- 快速入门：六步学习闭环（步骤卡片，需求 2） ----
    st.subheader("快速入门")
    steps = [
        ("1", "AI 自主规划",
         "无需上传课件，输入任意主题（如：量子力学基础），"
         "AI 自动构建知识图谱并生成学习路径。"),
        ("2", "上传课件",
         "在「上传解析」页上传 PDF / Word / TXT / Markdown 课件（单文件最大 20MB）。"
         "AI 会自动提取文本块、抽取知识点并构建专属知识库。"),
        ("3", "智能问答",
         "在「智能问答」页针对课件内容提问，回答自动标注来源页码与片段，"
         "支持多轮追问与历史会话管理。"),
        ("4", "学习诊断",
         "在「学习诊断」页让 AI 按知识点出选择题，作答后生成结构化诊断报告，"
         "精准定位薄弱知识点。"),
        ("5", "路径推荐",
         "根据诊断出的薄弱点，结合知识图谱先修依赖，自动生成有序的学习路径。"),
        ("6", "评估反馈",
         "完成前测—复习—后测，用学习增益（ALG）量化进步幅度，"
         "并可导出 SVG / Markdown 报告留存分享。"),
    ]
    for num, title, desc in steps:
        # 单行拼接（无缩进/空行）——原因见 _empty_card 的 Markdown 代码块泄露注释
        st.markdown(
            f'<div style="display:flex; align-items:flex-start; gap:12px; padding:12px 16px;'
            f' margin:6px 0; border-radius:14px;'
            f' background:linear-gradient(180deg,#ffffff,#f7f8fa);'
            f' border:1px solid rgba(0,0,0,.05);">'
            f'<div style="min-width:28px; height:28px; border-radius:50%; background:#3478f6;'
            f' color:#fff; display:flex; align-items:center; justify-content:center;'
            f' font-weight:600; font-size:.95rem;">{num}</div>'
            f'<div style="line-height:1.5;"><b>{title}</b><br>'
            f'<span style="color:#86868b; font-size:.9rem;">{desc}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption("提示：左侧导航可随时在各功能页之间切换；解析完成前请勿刷新页面。")

    st.divider()

    # ---- 常见问题 FAQ（折叠面板，需求 2） ----
    st.subheader("常见问题")
    faqs = [
        ("支持哪些文件格式？",
         "目前支持 **PDF、Word（.docx）、TXT、Markdown（.md）** 四类课件格式，"
         "单文件最大 **20MB**。扫描版（纯图片）PDF 无法提取文字，建议使用文字版课件。"),
        ("解析失败怎么办？",
         "① 确认文件未加密、不是纯图片扫描件；\n\n"
         "② 检查网络连接后，在「上传解析」页重新点击开始解析；\n\n"
         "③ 文件过大时请压缩或拆分后重新上传；\n\n"
         "④ 若仍失败，请联系管理员在「任务监控」中查看具体失败原因。"),
        ("如何切换账号？",
         "点击侧边栏底部的「退出登录」，回到登录页后输入另一账号的昵称与密码登录即可。"
         "退出时本账号的学习会话与文档缓存会全部安全清空，两个账号的数据互相隔离、绝不串号。"),
        ("如何重置密码？",
         "请联系管理员为您的账号重置密码。密码采用加盐哈希加密存储，"
         "任何人（包括管理员）都无法查看原密码，只能重新设置。"),
        ("我没有课件，可以学习吗？",
         "可以。在首页点击「AI 自主规划」，输入你想学的主题即可。"),
        ("AI 回答的依据是什么？",
         "回答完全基于您上传的课件内容检索生成，并自动标注来源页码与原文片段；"
         "如果课件中没有相关内容，AI 会如实说明，不会凭空编造。"),
        ("每分钟能发起多少次 AI 请求？",
         "为保障服务稳定，单用户每分钟最多 **5 次** AI 请求"
         "（问答、出题、报告、图谱构建均计入）。超限时页面会提示您休息一分钟再试。"),
        ("我的数据是私密的吗？",
         "课件原件按账号**物理隔离**存储（每人独立一个 data/账号名/ 目录），"
         "问答、诊断等学习记录也仅本人可见；管理员仅能查看统计信息与文件清单，不能操作您的数据。"),
        ("上传的文件可以删除吗？",
         "可以。在「上传解析」页的「文件管理」区点击「删除」，"
         "确认后课件原件与关联的问答/诊断/路径/评估记录会被一并安全删除。"),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(a)

    st.divider()

    # ---- 法律条款与隐私（需求：隐私与版权合规——登录页复选框所指的协议全文） ----
    st.subheader("用户协议与隐私政策")
    with st.expander("《用户协议》"):
        st.markdown(
            "欢迎使用智学 AI 学习助手。使用本系统前，请您仔细阅读并理解本协议：\n\n"
            "1. **服务性质**：本系统基于您上传的课件，提供智能问答、学情诊断、"
            "学习路径推荐等 AI 辅助学习功能。\n\n"
            "2. **账号规则**：您需使用真实、合规的昵称注册账号，并妥善保管密码；"
            "账号下的所有学习行为均视为您本人操作。\n\n"
            "3. **内容规范**：您上传的文档应为您拥有合法版权或使用权的内容，"
            "严禁上传涉密、侵权、色情、暴力或其他违法违规内容；"
            "由此产生的一切责任由上传者自行承担。\n\n"
            "4. **AI 生成内容**：AI 生成的回答、诊断与建议均基于您上传的课件生成，"
            "仅供参考学习使用，不构成专业意见，请自行核实后再做判断。\n\n"
            "5. **合理使用**：请遵守系统的频次限制，避免滥用或以技术手段干扰系统正常运行。"
        )
    with st.expander("《隐私政策》"):
        st.markdown(
            "本系统尊重用户隐私，您上传的文档仅保存在您的独立存储空间，"
            "不会用于训练大模型，也不会被其他用户共享。\n\n"
            "1. **独立存储**：每位用户的课件保存在服务器的个人专属目录（data/账号名/）中，"
            "与其他用户物理隔离，任何其他用户都无法访问。\n\n"
            "2. **学习记录**：问答、诊断、评估等学习记录仅您本人可见；"
            "管理员仅能查看统计信息与文件清单，不会查看您的具体对话内容。\n\n"
            "3. **密码安全**：密码采用加盐哈希加密存储，"
            "任何人（包括管理员）都无法查看或还原您的原密码。\n\n"
            "4. **不用于训练**：您上传的文档与对话内容不会用于训练任何大语言模型。\n\n"
            "5. **删除权**：您可随时在「上传解析」页删除课件，"
            "课件原件与关联的学习记录将被一并安全删除。"
        )

    st.divider()
    st.caption("没有找到答案？请联系管理员，或在「系统日志」（管理员端）反馈问题现象。")


# ========== 学生端页面 0：首页（Coze 风格极简着陆页） ==========
def _goto_page(page_name):
    """首页快捷卡片「开始使用/开始规划」的跳转回调。
    on_click 回调先于下一轮 widget 实例化执行，此处写 st.session_state.page 安全
    （在脚本主体里直接写会报 "cannot be modified" 异常）。"""
    st.session_state.page = page_name


def _sync_nav_from_radio():
    """侧边栏导航（受控 radio）的 on_change 回调：用户切换菜单时同步到 page。
    与 _goto_page / restore_history 等回调写 page 的机制并存，互不干扰；
    仅在用户真实操作导航时触发（程序化跳转走各自回调）。"""
    st.session_state.page = st.session_state["nav_choice"]


def page_home():
    """
    学生端首页（登录后的默认着陆页）：Coze 官网式极简、现代风格。
    全部使用 Streamlit 原生组件（st.markdown / st.columns / st.container(border=True)
    / st.dataframe），不注入自定义 CSS（仅内联 text-align 实现标题居中）。
    结构：
      1. 顶部问候语：居中大标题 + 居中副标题；
      2. 快捷操作区：三张大卡片（上传解析 / 智能问答 / 学习诊断），
         卡片底部「开始使用」经 on_click 回调跳转对应功能页；
      3. 最近学习记录：当前用户最近 5 个课件的问答/诊断次数一览（dataframe）。
    """
    # ---- 1. 顶部问候语（居中） ----
    st.markdown("<h1 style='text-align:center;'>嗨，我是 智学引擎 </h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#7F8C8D; font-size:1.2rem;'>"
        "今天想从哪里开始呢？</p>",
        unsafe_allow_html=True)

    # ---- 2. 快捷操作区（四张大卡片，2×2 两行布局，原生 border 容器） ----
    # (标题, 描述, 跳转目标页, 按钮文案)——跳转目标必须是页面注册表 STUDENT_PAGES 的确切键名
    cards = [
        ("上传解析", "上传课件PDF/Word，自动构建知识图谱", "上传解析", "开始使用"),
        ("智能问答", "基于文档的多轮对话，引用原文溯源", "智能问答", "开始使用"),
        ("学习诊断", "KCR三维认知诊断，量化学习增益", "学习诊断", "开始使用"),
        ("AI 自主规划", "输入任意主题，自动构建知识图谱与学习路径，无需上传课件",
         "AI 自主规划", "开始规划"),
    ]
    # 两行两列：4 卡对称分布，避免单行四列的窄卡拥挤。
    # 卡内首元素用 st.html 注入 home-card-{n} 锚点（st.markdown 会把标题 id
    # 重写为随机锚点 id，导致 CSS :has 定位失效——见全局样式区"首页卡矩阵"）。
    card_no = 0
    for row_idx, row_cards in enumerate((cards[:2], cards[2:])):
        cols = st.columns(2)
        for col_idx, (col, (title, desc, target, btn_label)) in enumerate(zip(cols, row_cards)):
            with col, st.container(border=True):
                st.html(f'<span id="home-card-{card_no}"></span>')
                st.markdown(f"#### {title}")
                st.caption(desc)
                st.button(btn_label, key=f"home_go_{card_no}",
                          width="stretch",
                          disabled=st.session_state.parsing,
                          on_click=_goto_page, args=(target,))
                card_no += 1

    # ---- 3. 最近学习记录（当前用户最近 5 个课件的学习活动统计） ----
    st.divider()
    with st.container(border=True):
        st.markdown("##### 最近学习记录")
        import pandas as pd   # 局部导入：仅本页需要（与 page_admin_logs 惯例一致）
        # 按课件分组取最近 5 个（问答/诊断/路径/评估记录都在组内，这里展示前两类次数）
        groups = get_history_grouped(st.session_state.user_name,
                                     st.session_state.role, max_files=5)
        if not groups:
            st.info("还没有学习记录——点上方「开始使用」上传第一份课件吧！")
        else:
            rows = [{
                "课件名称": g["filename"],
                "问答次数": len(g["qa"]),
                "诊断次数": len(g["diagnosis"]),
                "最近活动": str(g["latest_time"])[:16],
            } for g in groups]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("问答/诊断次数为该课件下的累计记录条数；"
                       "侧边栏「最近学习记录」可展开恢复具体内容。")


# ========== 学生端页面 0.5：AI 自主规划（无课件主题规划） ==========
def page_topic_planner():
    """
    AI 自主规划：无需上传课件，输入学习主题，AI 直接生成知识图谱与学习路径。
    全流程（点击「开始规划」后）：
      TopicPlannerAgent.plan 单次 AI 调用生成大纲（知识点 + 先修依赖）→
      normalize 规范化（幻觉边过滤）→ build_graph 组装 DiGraph →
      plan_path 分层拓扑排序 → 构造"虚拟文件"存入 files 表（归属当前用户）→
      灌入 doc/index/retriever 上下文 → 结果展示 + 去诊断/去评估快捷跳转。
    虚拟文件约定：filename = "主题：{topic}"；data.chunks = 每个知识点一个文本块
    （pages 为空数组），使诊断出题/问答引用/历史恢复等下游功能全部兼容。
    """
    import pandas as pd

    st.title("AI 自主规划学习路径")
    st.caption("输入你想学习的主题或话题，无需上传课件，系统将自动为你构建知识图谱与学习路径。")

    # ---- 规划入口：主题输入 + 一键规划 ----
    with st.container(border=True):
        topic = st.text_input(
            "学习主题",
            placeholder="例如：Transformer架构、量子力学基础、中国近代史...",
            max_chars=60,
            help="输入一个学科主题或具体话题，AI 将生成 8-12 个核心知识点及其先修关系",
        )
        if st.button("开始规划", type="primary", width="stretch",
                     disabled=st.session_state.parsing):
            # ---- 前置校验：主题非空 + Agent 开关 + AI 限流 ----
            if not topic.strip():
                st.warning("请先输入想学习的主题。")
                return
            if not get_agent_config()["planner_enabled"]:
                agent_disabled_msg("「自主规划 Agent」")
                return
            if not ai_rate_limit_check():
                return

            # ---- 任务追踪 + 限时 AI 调用（与诊断出题同一套超时/降级机制） ----
            agent = TopicPlannerAgent()
            task_id = create_task("topic_plan",
                                  st.session_state.user_name, st.session_state.role)
            update_task_status(task_id, "processing")
            data = call_agent_limited(
                lambda: agent.plan(topic.strip()),
                agent=agent, timeout=90,
                spinner_text=f"AI 正在为「{topic.strip()}」构建学习大纲...")
            if data is None:
                update_task_status(task_id, "failed",
                                   (agent.last_error or {}).get("message", "规划失败"))
                ai_fail_hint(agent)
                return

            # ---- 规范化 + 图谱 + 路径（本地计算，零 API） ----
            try:
                kps, deps = TopicPlannerAgent.normalize(data)
            except ValueError as e:
                update_task_status(task_id, "failed", f"AI 返回格式异常：{e}")
                st.error("AI 返回的大纲格式异常，请重试一次。")
                return
            if len(kps) < 3:
                update_task_status(task_id, "failed", "知识点数量不足")
                st.warning("AI 只提取出不足 3 个知识点，无法构成学习路径，请更换表述或重试。")
                return
            graph = TopicPlannerAgent.build_graph(kps, deps)
            path = TopicPlannerAgent.plan_path(graph)

            # ---- 虚拟文件落库：每个知识点一个文本块（供诊断出题/问答引用/历史恢复） ----
            chunk_list = [{"id": f"tp{i:03d}", "pages": [],
                           "text": f"知识点：{k['name']}。{k['description']}"
                           if k["description"] else f"知识点：{k['name']}"}
                          for i, k in enumerate(kps, start=1)]
            file_id = save_file(
                f"主题：{topic.strip()}",
                data={
                    "chunks": chunk_list,
                    "knowledge_candidates": kps,
                    "graph": graph_to_dict(graph),
                    "topic_plan": {"topic": topic.strip(), "path": path},
                },
                user_name=st.session_state.user_name, role=st.session_state.role,
            )

            # ---- 灌入当前学习上下文：诊断/评估/问答/图谱页立即可用（与解析完成等效） ----
            doc = st.session_state.doc
            doc["filename"] = f"主题：{topic.strip()}"
            doc["chunks"] = chunk_list
            doc["knowledge_candidates"] = kps
            doc["graph"] = graph
            st.session_state.file_id = file_id
            # 本地重建 TF-IDF 检索索引（零 API，与 load_doc_from_db 同源逻辑）
            index = IndexAgent().build([{"id": c["id"], "text": c["text"]} for c in chunk_list])
            st.session_state.index = index
            st.session_state.retriever = RetrieverAgent(index)
            # 结果数据（跨 rerun 展示）；清掉旧问答会话（课件已切换，会话按课件隔离）
            st.session_state.planner_result = {"topic": topic.strip(), "kps": kps, "path": path}
            st.session_state.qa_session_id = None
            st.session_state.qa_session_name = None
            st.session_state.chat_history = []

            update_task_status(task_id, "success")
            st.toast("学习规划已生成，已载入为当前学习内容")
            st.rerun()

    # ---- 结果展示（规划完成后跨 rerun 常驻，直到下次规划覆盖） ----
    result = st.session_state.get("planner_result")
    if not result:
        # 未规划过：展示三步流程说明，页面不留空白
        with st.container(border=True):
            st.markdown("##### 规划流程")
            st.markdown(
                "1. 输入主题，AI 一次性生成 8-12 个核心知识点与先修依赖关系；\n"
                "2. 系统自动组装知识图谱，并按拓扑排序生成由浅入深的学习路径；\n"
                "3. 规划结果自动保存为你的学习内容，可直接进行问答、诊断与评估。"
            )
        return

    st.divider()
    st.success(f"已为「{result['topic']}」生成学习规划："
               f"{len(result['kps'])} 个知识点 · {len(result['path'])} 步学习路径"
               f"（已载入为当前学习内容，问答/诊断/评估/图谱均可直接使用）")

    # ---- 知识点列表（表格：序号 / 知识点 / 难度 / 简介） ----
    st.subheader("核心知识点")
    kps_df = pd.DataFrame([
        {"序号": i, "知识点": k["name"], "难度": k["difficulty"], "简介": k["description"]}
        for i, k in enumerate(result["kps"], start=1)
    ])
    st.dataframe(kps_df, width="stretch", hide_index=True)

    # ---- 推荐学习路径（分层拓扑序：阶段 / 顺序 / 知识点 / 理由） ----
    st.subheader("推荐学习路径")
    path_df = pd.DataFrame([
        {"阶段": f"第 {s['stage']} 阶段", "顺序": s["order"],
         "知识点": s["knowledge_point"], "建议理由": s["reason"]}
        for s in result["path"]
    ])
    st.dataframe(path_df, width="stretch", hide_index=True)
    st.caption("同阶段的知识点互不依赖、可并行学习；跨阶段必须先完成前一阶段。")

    # ---- 快捷跳转：诊断 / 评估（虚拟文件已是当前上下文，两页可直接使用） ----
    c1, c2 = st.columns(2)
    c1.button("去诊断（测试当前掌握程度）", type="primary", width="stretch",
              disabled=st.session_state.parsing,
              on_click=_goto_page, args=("学习诊断",))
    c2.button("去学习评估", width="stretch",
              disabled=st.session_state.parsing,
              on_click=_goto_page, args=("学习评估",))


# ========== 页面注册表（侧边栏导航 -> 页面函数） ==========
# 学生端：学习全流程（"我的用量"复用 page_usage，按当前用户过滤）
STUDENT_PAGES = {
    "首页": page_home,
    "上传解析": page_upload,
    "AI 自主规划": page_topic_planner,
    "智能问答": page_chat,
    "知识图谱": page_graph,
    "学习诊断": page_diagnosis,
    "路径推荐": page_path,
    "学习评估": page_eval,
    "我的用量": page_usage,
    "个人中心": page_profile,
    "历史记录": page_history,
    "帮助中心": page_help,
}
# 导航隐藏页：注册在 STUDENT_PAGES（路由可达、query 参数可恢复），但不出现在
# 侧边栏菜单——仅经首页卡片「开始规划」等入口写入 session_state.page 进入，
# 保持左侧导航的克制（Vercel 式：导航只放高频主路径）。
HIDDEN_PAGES = {"AI 自主规划"}
# 开发者端：运营与管理视角（退出登录为侧边栏底部按钮，非页面）
ADMIN_PAGES = {
    "数据总览": page_admin_overview,
    "用户管理": page_admin_users,
    "全平台用量": page_admin_usage,
    "任务监控": page_admin_tasks,
    "系统告警": page_admin_alerts,
    "系统维护": page_admin_maintenance,
    "系统监控": page_admin_monitor,
    "Agent 配置": page_admin_agents,
    "系统日志": page_admin_logs,
}


# ========== 主程序 ==========
def main():
    from migrate import run_migrations   # 数据库版本化迁移（PRAGMA user_version）
    run_migrations()
    init_db()

    # API Key 检查（需求 3）：缺失时显示友好的配置引导页，不渲染任何功能、不崩溃
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        error_logger.error("启动检查：DEEPSEEK_API_KEY 未配置")
        st.error("AI 服务尚未配置，应用无法启动")
        st.markdown(
            """
            ### 请按以下步骤完成配置（约 1 分钟）：

            1. 前往 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并创建 API Key
            2. 在项目根目录找到（或新建）`.env` 文件
            3. 在文件中写入一行：`DEEPSEEK_API_KEY=你的密钥`
            4. 保存后重启本应用即可正常使用

            > 配置信息只保存在本机 `.env` 文件中，不会被上传或分享。
            """
        )
        if st.button("我已配置完成，重新检查", type="primary"):
            st.rerun()
        st.stop()

    # 会话状态默认值（只在首次运行时创建）
    defaults = {
        "doc": {"filename": None, "chunks": [], "knowledge_candidates": [], "graph": None},
        "file_id": None,     # 当前课件在 files 表中的记录 id（各页面写记录时关联）
        "index": None,       # IndexAgent 实例（含 numpy 矩阵）
        "retriever": None,   # RetrieverAgent 实例（封装索引）
        "fp": None,          # 上传文件指纹（文件名+大小），避免重复解析
        "quiz": None,        # 当前诊断题目
        "report": None,      # 当前诊断报告
        "path": None,        # 当前学习路径
        "planner_result": None,  # AI 自主规划结果 {"topic", "kps", "path"}（跨 rerun 展示）
        "evaluation": None,  # 学习增益评估结果 {"alg", "pre_score", "post_score", "full_score", "report"}
        "pre_quiz": None,    # 前测题目
        "pre_score": None,   # 前测得分
        "post_quiz": None,   # 后测题目（与前测平行的卷子）
        "post_score": None,  # 后测得分
        "last_qa": None,     # 最近一次问答 {"question", "answer", "contexts", "knowledge_points"}
        "chat_history": [],  # 问答页多轮对话 [{"role", "content", "contexts", "knowledge_points"}]
        # ---- 智能问答多会话管理 ----
        "qa_session_id": None,    # 当前会话 id（None=尚未开话，首条问答入库时才落库）
        "qa_session_name": None,  # 当前会话名称（首条提问自动取问题前 12 字命名）
        "history_warning": None,  # 历史恢复时的提示（如"原始文件已丢失"），显示一次即清除
        "user_name": None,          # 当前用户昵称（未登录为 None，登录页写入）
        "role": None,               # 当前角色（student=学习端 / admin=开发者端，未登录为 None）
        "last_role": None,          # 上一次运行时的角色：变化时触发业务数据清空（双端隔离）
        "denied_msg": None,         # 越权访问提示（学生端访问开发者端页面时显示一次）
        "dialog_file": None,     # 文件删除确认弹窗的目标文件信息（id/文件名/关联数/物理路径）
        # ---- 上传解析流水线状态（解决"解析中切换页面导致中断/状态丢失"问题） ----
        "uploaded_file": None,    # 持久化的上传文件对象（重跑间稳定，与 uploader 状态解耦）
        "uploaded_fp": None,      # 缓存的文件指纹（文件名+大小）
        "store_paths": {},        # 文件名 -> 原件物理存储相对路径（data/{user}/{filename}，上传即落盘）
        "parsing": False,         # 解析进行中（True 时禁用侧边栏导航，防止脚本被中断）
        "parse_stage": 0,         # 流水线断点：0=空闲，1=切块 2=知识点 3=索引 4=图谱 5=收尾
        "graph_ok": False,        # 流水线步骤 4 的图谱构建结果（收尾时生成提示用）
        "graph_prev": None,       # 上一份课件的图谱（单文件上传时用于增量更新提示，一次性）
        "graph_merge_pending": False,  # 新文档解析完成且已有图谱：待用户选择增量/全新/跳过
        "parse_error": None,      # 解析错误信息（显示一次即清除）
        "parse_result_msg": None, # 解析完成消息 (kind, text)（显示一次即清除）
        "kp_extract_failed": False,  # 知识点抽取全失败的降级标志（True 时跳过图谱构建并提示）
        "ai_call_times": [],      # AI 请求时间戳列表（轻量限流：每分钟最多 5 次，需求 2）
        "task_id": None,          # 当前解析任务的 tasks 表 id（任务追踪生命周期用）
        # ---- 多文件批量解析状态（队列式：一个文件解析完成自动衔接下一个） ----
        "upload_queue": [],       # 待解析的剩余文件列表（UploadedFile 对象）
        "upload_total": 0,        # 本批次文件总数（0 表示单文件模式）
        "batch_results": [],      # 批次结果 [{"filename", "ok", "info"}]
        # ---- Agent 运行配置（开发者端"Agent 配置"页修改，执行时实时读取） ----
        "agent_config": dict(DEFAULT_AGENT_CONFIG),
        "login_token": None,      # 持久登录令牌（挂 URL ?t=，刷新浏览器后凭它恢复登录）
    }
    # ---- 登出清理（必须发生在任何 widget 实例化之前） ----
    # 侧边栏"退出登录"按钮只设置 logout_pending 标志并触发重跑；
    # 真正的键删除放到这里：此时 radio（key="page"）尚未渲染，
    # 直接删除不会触发 "widget key cannot be modified" 异常。
    if st.session_state.get("logout_pending"):
        del st.session_state["logout_pending"]
        # 先删持久登录令牌（数据库行 + URL 参数），登出后刷新浏览器不再自动恢复登录
        _tok = st.session_state.get("login_token")
        if _tok:
            delete_login_session(_tok)
            st.query_params.pop("t", None)
        for k in list(st.session_state.keys()):
            if not k.startswith("_"):   # 下划线开头的是 Streamlit 框架内部键，不可删
                if k == "agent_config":
                    # 保留 Agent 配置（系统级配置，非业务数据）：管理员在"Agent 配置"页
                    # 修改后，同一浏览器切换到学习端登录仍能读取，保证"配置 -> 执行"闭环
                    continue
                del st.session_state[k]
        st.rerun()

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---- 登录态恢复：浏览器刷新会新建服务端会话（session_state 全部清空）， ----
    # ---- 凭 URL 中挂着的持久令牌（?t=xxx）查库恢复身份，实现"刷新不掉线" ----
    if not st.session_state.user_name:
        _tok = st.query_params.get("t")
        if _tok:
            _sess = get_login_session(_tok)
            if _sess and not is_user_disabled(_sess["user_name"]):
                st.session_state.user_name = _sess["user_name"]
                st.session_state.role = _sess["role"]
                st.session_state.login_token = _tok

    # ---- 禁用账号实时生效：管理员禁用后，该用户的下一次交互即被强制下线 ----
    # （每轮脚本运行都检查一次 users.is_disabled；logout 复位身份，业务数据一并清空）
    if st.session_state.user_name and is_user_disabled(st.session_state.user_name):
        logout()
        reset_business_state()
        # 同步作废持久令牌并清掉 URL 参数，防止刷新后又被令牌拉回
        _tok = st.session_state.get("login_token")
        if _tok:
            delete_login_session(_tok)
            st.query_params.pop("t", None)
        st.warning("该账号已被管理员禁用，如有疑问请联系管理员。")
        page_login()
        return

    # ---- 登录态持久化：会话有身份但还没有令牌（刚登录/刚注册）， ----
    # ---- 签发令牌写入数据库并挂到 URL，之后的浏览器刷新即可自动恢复登录 ----
    if st.session_state.user_name and not st.session_state.login_token:
        _new = create_login_session(st.session_state.user_name, st.session_state.role)
        st.session_state.login_token = _new
        st.query_params["t"] = _new

    # ---- 双端分离入口：未登录时只显示登录页（学生/管理员两个入口 Tab） ----
    if not st.session_state.user_name:
        page_login()
        return

    role = st.session_state.role
    pages = ADMIN_PAGES if role == "admin" else STUDENT_PAGES   # 按角色渲染不同导航

    # ---- 导航键合法性校验 + 越权拦截 + 角色切换隔离 + 页面恢复 ----
    # session["page"] 是 radio 的 widget key：若残留了另一个角色的页面名
    # （如从学生端切到管理员端后残留"上传解析"），该值不在当前角色的
    # 选项列表里，radio 会直接抛异常白屏。渲染前做四件事（此时 widget
    # 尚未实例化，修改 session 值是安全的）：
    #   1) 凭 URL 的 page 参数恢复页面（仅当会话内页面尚未渲染过——即 F5
    #      刷新后的全新服务端会话；正常导航时 URL 参数滞后一个交互，
    #      若宽放条件会误判"点击回首页"为刷新而把用户拽回旧页面）；
    #   2) 学生端访问开发者端页面 -> 记录"权限不足"提示，跳回学习端首页；
    #   3) 其他非法值 -> 静默复位到当前端首页；
    #   4) 角色发生变化（student<->admin）-> 清空全部业务数据，双端完全隔离。
    cur_page = st.session_state.get("page")
    qp_page = st.query_params.get("page")
    if (qp_page and qp_page != cur_page and qp_page in pages
            and cur_page is None):   # None = F5 刷新后的首次渲染（新会话无 page 值）
        cur_page = qp_page   # 需求 2：F5 刷新后自动回到刷新前所在的页面
        st.session_state.page = qp_page
    if cur_page not in pages:
        # 非管理员访问开发者端页面（会话残留 or URL ?page= 伪造）-> 提示 + 跳回学生端首页
        if role != "admin" and (cur_page in ADMIN_PAGES or qp_page in ADMIN_PAGES):
            st.session_state.denied_msg = "权限不足：开发者端页面仅限管理员访问，已返回学习端首页。"
        st.session_state.page = list(pages.keys())[0]
    if st.session_state.last_role != role:
        reset_business_state()
        st.session_state.last_role = role

    # ---- 会话恢复（需求 2）：刷新后凭 URL 的 sid 参数还原会话与课件上下文 ----
    # 仅在"当前没有任何打开的会话且课件未加载"（= 刚刷新的初始态）时尝试恢复；
    # 归属校验由 get_qa_session_meta 强制（他人会话按不存在处理，防串号）
    qp_sid = st.query_params.get("sid")
    if (qp_sid and not st.session_state.qa_session_id
            and not st.session_state.doc["chunks"] and role != "admin"):
        meta = get_qa_session_meta(qp_sid, st.session_state.user_name, st.session_state.role)
        if meta:
            # 先恢复会话所属的课件上下文（文本块 + 检索索引），再恢复对话气泡；
            # 课件已被删除则无法恢复，清理失效参数
            if meta.get("file_id") and load_doc_from_db(meta["file_id"]):
                _open_qa_session(qp_sid)
                st.toast("已恢复上次会话与课件上下文")
            else:
                st.query_params.pop("sid", None)
        else:
            st.query_params.pop("sid", None)

    # ---- 侧边栏：用户卡 + 按角色的导航 + 学生专属组件 + 底部退出 ----
    with st.sidebar:
        st.title("智学 AI 助手")

        # ---- 用户身份卡 ----
        if role == "admin":
            st.success(f"**{st.session_state.user_name}**\n\n开发者端")
        else:
            st.success(f"**{st.session_state.user_name}**\n\n学习端")

        # 解析进行中禁用导航切换：防止新的交互中断解析脚本导致解析失败
        # 受控导航：options 过滤掉导航隐藏页（HIDDEN_PAGES，如"AI 自主规划"——
        # 仅经首页卡片等入口写 session_state.page 进入，保持菜单克制）。
        # page（真实页面）可指向隐藏页：此时 radio 高亮回落到首页，页面分发
        # 仍渲染真实页面。两个同步机制并存：
        #   ① 用户点击菜单 -> on_change 回调把选择写入 page；
        #   ② 程序化跳转（历史恢复/首页卡片等回调）写 page 后，此处强制
        #      nav_choice 对齐（radio 实例化前写 widget key 合法），保证高亮一致。
        visible_pages = [k for k in pages if k not in HIDDEN_PAGES]
        nav_cur = (st.session_state.page
                   if st.session_state.page in visible_pages else visible_pages[0])
        # 注意：初始选中值通过实例化前写入 session_state 实现（受控），
        # 不能再传 index 参数——两者并存会触发 Streamlit 冲突告警
        st.session_state["nav_choice"] = nav_cur
        st.radio("功能导航", options=visible_pages,
                 key="nav_choice", on_change=_sync_nav_from_radio,
                 disabled=st.session_state.parsing)
        # 真实页面以 session_state.page 为准（radio 返回值在隐藏页时是回落项）；
        # page 变量语义与旧版一致：后续 query 同步、页面分发均使用
        page = st.session_state.page
        # 需求 3：地址栏 page 参数与当前页面实时同步（刷新后据此恢复页面）
        st.query_params["page"] = page
        if st.session_state.parsing:
            stage = min(st.session_state.parse_stage, 4)
            st.warning(f"正在解析课件（步骤 {stage}/4），请勿切换页面")

        # ---- 学生端专属组件：知识库状态 + 问答会话 + 最近学习记录 ----
        if role != "admin":
            doc = st.session_state.doc
            st.divider()
            if doc["chunks"]:
                st.caption(f"当前知识库：**{doc['filename']}**（{len(doc['chunks'])} 个文本块）")

                # ---- 智能问答多会话管理：新建 / 历史列表 / 重命名 / 删除 ----
                # 会话按当前课件隔离（qa 记录绑定 file_id），切换课件后显示对应会话列表
                st.caption("智能问答会话")
                # on_click 回调模式：回调内完成"新建会话 + 跳转问答页"，执行完自动整页重跑
                st.button("新建对话", use_container_width=True,
                          disabled=st.session_state.parsing,
                          on_click=_click_new_session,
                          help="开始一段新对话（首条提问后自动保存进历史列表）")
                sessions = list_qa_sessions(st.session_state.user_name, st.session_state.role,
                                            st.session_state.file_id)
                # 新建后尚未提问的会话不在数据库里（懒落库设计）——
                # 在列表顶部补一条占位记录，否则"新建对话"点击后在界面上毫无变化
                if (st.session_state.qa_session_id
                        and not any(s["session_id"] == st.session_state.qa_session_id
                                    for s in sessions)):
                    sessions = [{"session_id": st.session_state.qa_session_id,
                                 "session_name": st.session_state.qa_session_name or "新对话",
                                 "count": 0, "last_time": None}] + sessions
                if not sessions:
                    st.caption("暂无历史会话")
                for i, s in enumerate(sessions):
                    is_cur = s["session_id"] == st.session_state.qa_session_id
                    # 行式布局：名称占宽列（加粗、长名省略号），右侧两个极小操作按钮
                    # （CSS .sb-session-* 控制：灰字无框、white-space:nowrap 防折行、等高对齐）
                    row = st.columns([3, 1, 1])
                    # 当前会话用 primary 按钮样式区分（侧边栏 CSS 覆写为浅绿底深绿字）
                    label = str(s["session_name"] or "未命名会话")
                    # 占位会话（未落库）没有 last_time，用专属提示文案
                    help_txt = (f"{s['count']} 条问答 · 最后提问 {str(s['last_time'])[:16]}"
                                if s.get("last_time") else "新对话 · 首条提问后自动保存")
                    # on_click 回调先于 widget 实例化执行：回调内写 page 跳转页面是安全的
                    # （直接在脚本主体写 st.session_state.page 会报 "cannot be modified" 异常）
                    row[0].button(
                        label, key=f"qa_s_{i}", use_container_width=True,
                        disabled=st.session_state.parsing,
                        help=help_txt,
                        type="primary" if is_cur else "secondary",
                        on_click=_open_qa_session, args=(s["session_id"],),
                    )
                    if row[1].button("改名", key=f"qa_r_{i}", disabled=st.session_state.parsing,
                                     help="重命名会话"):
                        # 先清掉重命名输入框的上次输入，再弹确认框（避免残留旧值）
                        st.session_state.pop(f"rename_{s['session_id']}", None)
                        st.session_state.dialog_sid = s["session_id"]
                        st.session_state.dialog_sname = s["session_name"]
                        rename_session_dialog()
                    if row[2].button("删除", key=f"qa_d_{i}", disabled=st.session_state.parsing,
                                     help="删除会话"):
                        st.session_state.dialog_sid = s["session_id"]
                        st.session_state.dialog_sname = s["session_name"]
                        delete_session_dialog()
            else:
                st.caption("当前知识库：未设置")

            st.divider()
            st.markdown('<p class="sb-sec-title">最近学习记录</p>', unsafe_allow_html=True)
            st.caption("按课件归档，点击展开")
            # 按课件分组拉取最近 5 个课件的学习活动（utils/db.get_history_grouped）
            groups = get_history_grouped(st.session_state.user_name,
                                         st.session_state.role, max_files=5)
            if not groups:
                st.caption("暂无记录")
            # 每个课件一个折叠区（默认收起，保持侧边栏整洁）：
            # 展开后按 问答→诊断→路径→评估 顺序，每类只显示最新一条摘要；
            # 点击子项触发 restore_history 回调：恢复该记录完整内容并自动跳转对应页面
            for gi, g in enumerate(groups):
                n_total = sum(len(g[k]) for k in ("qa", "diagnosis", "path", "eval"))
                with st.expander(f"{g['filename']} · {n_total} 条", expanded=False):
                    st.caption(f"最近活动：{g['latest_time'][:16]}")
                    for cat, icon, cname in (("qa", "", "问答"),
                                             ("diagnosis", "", "诊断"),
                                             ("path", "", "路径"),
                                             ("eval", "", "评估")):
                        recs = g[cat]
                        if not recs:
                            continue
                        latest = recs[0]   # 组内按时间倒序，首条即最新一条
                        st.button(
                            f"{icon} {cname}：{_history_summary(latest)}",
                            key=f"hist_{gi}_{cat}", use_container_width=True,
                            disabled=st.session_state.parsing,
                            help=(f"共 {len(recs)} 条{cname}记录，点击恢复最新一条"
                                  f"（{latest['time'][5:16]}）并进入对应页面"),
                            on_click=restore_history, args=(latest,),
                        )

        st.divider()
        # ---- 侧边栏底部：帮助中心快捷入口（学生端，一键跳转） + 退出登录（双端通用） ----
        if role != "admin":
            if st.button("帮助中心", use_container_width=True,
                         disabled=st.session_state.parsing, on_click=_goto_help):
                pass   # 跳转由 on_click 回调完成
        if st.button("退出登录", use_container_width=True,
                     disabled=st.session_state.parsing):
            do_logout()

    # ---- 历史恢复提示（如"原始文件已丢失"，显示一次即清除） ----
    if st.session_state.get("history_warning"):
        st.warning(st.session_state.history_warning)
        st.session_state.history_warning = None

    # ---- 越权访问提示（显示一次即清除） ----
    if st.session_state.denied_msg:
        st.warning(st.session_state.denied_msg)
        st.session_state.denied_msg = None

    # ---- 页面分发（页面级兜底：任何未捕获异常都不允许红色 Traceback 出现在页面上） ----
    try:
        pages[page]()
    except Exception:
        print(f"[ERROR page] Unhandled exception on page: {page}")
        error_logger.error("页面渲染未捕获异常（page=%s）", page, exc_info=True)
        st.error("页面出现了意外情况，请刷新重试。")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise   # st.stop() 等正常退出不放行会破坏流程，需原样抛出
    except Exception:
        # 双保险：main() 主体（init_db / 侧边栏渲染等）的未知异常也优雅拦截
        print("[ERROR main] Unhandled exception in main()")
        error_logger.error("main() 主体未捕获异常", exc_info=True)
        st.error("应用出现了意外情况，请刷新页面重试。")
