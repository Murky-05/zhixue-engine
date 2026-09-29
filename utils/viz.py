# -*- coding: utf-8 -*-
"""
utils.viz —— 学习评估可视化工具（plotly 雷达图 + 零依赖 SVG 报告卡）
====================================================================
职责：把「知识点掌握度」数据渲染成两种载体：
    1. 页面交互图：plotly 雷达图（知识点 >= 3 时）；不足 3 个时由调用方
       改用 st.bar_chart（本模块提供 bar_chart_data 造数据）。
    2. 导出图片：纯 Python 手绘「评估报告卡」SVG——零第三方依赖、无需
       kaleido/orca，浏览器 / Word / PPT 可直接打开插入（矢量不失真）。

掌握度数据约定（由 app.py 的 collect_eval_mastery 聚合）：
    [{"name": "知识点名", "pct": 0-100 整数}, ...]

设计规范（Apple 风格）：系统蓝 #0A84FF 主色、浅灰网格、大量留白、
中文系统字体栈（PingFang SC / Microsoft YaHei），无外部资源引用。
"""

import math
from xml.sax.saxutils import escape

# ---------- Apple 风格配色 ----------
ACCENT = "#0A84FF"    # 系统蓝：数据主色（填充/描边/强调文字）
GRID = "#E5E5EA"      # 浅灰：网格与底条
TEXT = "#1D1D1F"      # 近黑：正文
MUTED = "#86868B"     # 灰：次要文字
BG = "#F5F5F7"        # 卡外背景
LEVEL_COLORS = {"高增益": "#30D158", "中等增益": "#FF9F0A", "低增益": "#FF453A"}
FONT = "PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif"


# ================= 页面交互图（plotly） =================
def radar_fig(mastery, height=420):
    """
    知识点掌握度雷达图（plotly，页面交互用）。
    :param mastery: [{"name", "pct"}]，pct ∈ 0-100
    :return: plotly.graph_objects.Figure；知识点 < 3 个时返回 None
             （雷达至少需要 3 个维度，由调用方降级为条形图）
    """
    if not mastery or len(mastery) < 3:
        return None
    import plotly.graph_objects as go

    # 首尾各放一次第一个点，让多边形闭合
    names = [m["name"] for m in mastery] + [mastery[0]["name"]]
    vals = [m["pct"] for m in mastery] + [mastery[0]["pct"]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=names, fill="toself",
        line=dict(color=ACCENT, width=2.5),
        fillcolor="rgba(10,132,255,0.25)",
        marker=dict(size=7, color=ACCENT),
        hovertemplate="%{theta}： %{r}%<extra></extra>",   # 悬浮显示「知识点： xx%」
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(range=[0, 100], showticklabels=False,
                            gridcolor=GRID, angle=90),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=13, color=TEXT)),
        ),
        showlegend=False,
        paper_bgcolor="white",
        font=dict(family=FONT),
        margin=dict(l=70, r=70, t=36, b=36),
        height=height,
    )
    return fig


def bar_chart_data(mastery):
    """知识点 < 3 个时的降级数据：dict（st.bar_chart 直接可用）"""
    return {m["name"]: m["pct"] for m in mastery}


def radar_png_bytes(fig, scale=2.0):
    """
    plotly 图 -> PNG 字节流（需 kaleido；未安装或渲染失败返回 None）。
    调用方拿到 None 时不显示 PNG 导出按钮，优雅降级为 SVG 报告卡。
    """
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception:
        return None


# ================= 导出图片（零依赖 SVG 报告卡） =================
def _polar_point(cx, cy, r, angle_deg):
    """极坐标 -> 直角坐标（angle_deg: -90 为正上方，顺时针递增）"""
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _radar_svg_group(mastery, cx, cy, r):
    """雷达图 SVG 片段：网格环 + 轴线 + 数据多边形 + 顶点 + 标签"""
    n = len(mastery)
    step = 360.0 / n
    parts = []

    # ---- 网格：4 层同心多边形（25/50/75/100）----
    for ring in (25, 50, 75, 100):
        pts = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (_polar_point(cx, cy, r * ring / 100, -90 + i * step)
                         for i in range(n))
        )
        parts.append(f'<polygon points="{pts}" fill="none" stroke="{GRID}" stroke-width="1"/>')

    # ---- 轴线 + 轴端知识点标签 + 顶点百分比标签 ----
    for i, m in enumerate(mastery):
        ang = -90 + i * step
        ex, ey = _polar_point(cx, cy, r, ang)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>')
        # 文字锚点按象限调整：右侧 start / 左侧 end / 正上正下 middle
        lx, ly = _polar_point(cx, cy, r + 44, ang)
        cosv = math.cos(math.radians(ang))
        anchor = "middle" if abs(cosv) < 0.35 else ("start" if cosv > 0 else "end")
        label = m["name"] if len(m["name"]) <= 7 else m["name"][:7] + "…"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" font-size="14" fill="{TEXT}">{escape(label)}</text>')
        px, py = _polar_point(cx, cy, r * m["pct"] / 100, ang)
        vx, vy = _polar_point(cx, cy, r * m["pct"] / 100 + 20, ang)
        parts.append(
            f'<text x="{vx:.1f}" y="{vy:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-size="11.5" font-weight="600" '
            f'fill="{ACCENT}">{m["pct"]}%</text>')

    # ---- 数据多边形（半透明填充）+ 顶点圆点 ----
    pts = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (_polar_point(cx, cy, r * m["pct"] / 100, -90 + i * step)
                     for i, m in enumerate(mastery))
    )
    parts.append(
        f'<polygon points="{pts}" fill="rgba(10,132,255,0.22)" '
        f'stroke="{ACCENT}" stroke-width="2.5" stroke-linejoin="round"/>')
    for i, m in enumerate(mastery):
        x, y = _polar_point(cx, cy, r * m["pct"] / 100, -90 + i * step)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{ACCENT}" '
            f'stroke="white" stroke-width="1.5"/>')
    return "\n".join(parts)


def _bars_svg_group(mastery, x, y, width):
    """知识点 < 3 个时的横条卡 SVG 片段（逐行：名称 + 底条 + 数据条 + 百分比）"""
    parts = []
    row_h = 56
    bar_w = width - 210   # 左侧留名称区 170px、右侧留百分比 40px
    for i, m in enumerate(mastery):
        top = y + i * row_h
        label = m["name"] if len(m["name"]) <= 9 else m["name"][:9] + "…"
        parts.append(
            f'<text x="{x}" y="{top + 19}" font-size="14" fill="{TEXT}">{escape(label)}</text>'
            f'<rect x="{x + 170}" y="{top + 6}" width="{bar_w}" height="14" '
            f'rx="7" fill="{GRID}"/>'
            f'<rect x="{x + 170}" y="{top + 6}" width="{max(bar_w * m["pct"] / 100, 8)}" '
            f'height="14" rx="7" fill="{ACCENT}"/>'
            f'<text x="{x + 170 + bar_w + 12}" y="{top + 19}" font-size="13.5" '
            f'font-weight="600" fill="{TEXT}">{m["pct"]}%</text>')
    return "\n".join(parts), len(mastery) * row_h


def radar_report_svg(mastery, meta):
    """
    生成「评估报告卡」SVG 字符串（零依赖，可直接作为图片保存/插入文档）。
    :param mastery: [{"name", "pct"}]；>=3 画雷达，<3 降级为横条卡，空列表仅出头部指标
    :param meta: 可选键 {"user", "filename", "pre", "post", "full", "alg", "level", "time"}
    :return: SVG 文本（utf-8）
    """
    has_chart = bool(mastery)
    use_radar = has_chart and len(mastery) >= 3

    # ---- 画布高度自适应：头部 232 + 图区 + 底部 64 ----
    if use_radar:
        r = 158
        chart_h = 2 * (r + 58)
    elif has_chart:
        chart_h = len(mastery) * 56 + 40   # 横条卡：每行 56 + 上下留白
    else:
        chart_h = 90
    W, H = 880, 232 + chart_h + 64
    card_h = H - 48
    cy = 232 + chart_h / 2   # 雷达圆心（居中于图区）

    level = meta.get("level")
    alg_color = LEVEL_COLORS.get(level, ACCENT)
    full = meta.get("full")

    # ---- 头部：标题 / 副标题 / 三枚指标胶囊 ----
    sub = " · ".join(p for p in (meta.get("user"), meta.get("filename")) if p)
    pills = []
    pill_data = [
        ("前测得分", f'{meta.get("pre", "-")}/{full}' if full else str(meta.get("pre", "-")), TEXT),
        ("后测得分", f'{meta.get("post", "-")}/{full}' if full else str(meta.get("post", "-")), TEXT),
        ("学习增益 ALG", f'{meta.get("alg", "-"):.2f}' if isinstance(meta.get("alg"), (int, float)) else "-",
         alg_color),
    ]
    pw, gap, px0 = 252, 38, 60
    for i, (lab, val, vc) in enumerate(pill_data):
        px = px0 + i * (pw + gap)
        pills.append(
            f'<rect x="{px}" y="160" width="{pw}" height="56" rx="16" fill="{BG}"/>'
            f'<text x="{px + 20}" y="183" font-size="12.5" fill="{MUTED}">{lab}</text>'
            f'<text x="{px + 20}" y="206" font-size="21" font-weight="700" fill="{vc}">{val}</text>')

    # ---- 图区（雷达 / 横条 / 空态）----
    if use_radar:
        chart = _radar_svg_group(mastery, W / 2, cy, r)
    elif has_chart:
        chart, _ = _bars_svg_group(mastery, 60, 232 + 20, W - 120)
    else:
        chart = (f'<text x="{W / 2}" y="{cy}" text-anchor="middle" '
                 f'font-size="14" fill="{MUTED}">暂无知识点掌握度数据</text>')

    footer = (f'知识点掌握度雷达 · {escape(meta.get("time") or "")} · 由智学 AI 助手生成')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="{BG}"/>
<rect x="24" y="24" width="{W - 48}" height="{card_h}" rx="28" fill="white"/>
<text x="60" y="88" font-size="28" font-weight="700" fill="{TEXT}">学习评估报告</text>
<text x="60" y="118" font-size="14.5" fill="{MUTED}">{escape(sub or "智学 AI 助手")}</text>
{''.join(pills)}
{chart}
<text x="{W / 2}" y="{H - 40}" text-anchor="middle" font-size="12" fill="{MUTED}">{footer}</text>
</svg>'''
