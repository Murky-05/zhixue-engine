# -*- coding: utf-8 -*-
"""
utils.graph_viz —— 知识图谱可视化工具
=======================================
把 networkx 知识图谱转成 pyvis 交互式 HTML 字符串，
配合 streamlit.components.v1.html 即可嵌入页面显示。

视觉约定：
    - 节点颜色 = 难度（绿:基础 / 橙:中等 / 红:困难 / 灰:未知）
    - 节点大小 = 度数（与越多知识点存在先修关联，节点越大 = 越核心）
    - 边      = 先修关系，箭头指向"后学"的知识点，悬浮显示学习顺序说明

交互增强（AI 内容可溯源）：
    传入 node_sources（知识点 -> 课件原文段落映射）后，点击节点会在画布
    右上角弹出详情框，展示该知识点对应的原文段落与页码出处。
    实现方式：在 pyvis 生成的 HTML 末尾注入一段自定义 JS（vis.js click
    事件 + 固定定位 overlay），在 Streamlit iframe 内部闭环，零新依赖。

使用示例：
    from utils.graph_viz import build_graph_html
    import streamlit.components.v1 as components
    html = build_graph_html(graph)                      # graph: networkx.DiGraph
    components.html(html, height=620, scrolling=False)  # 在 Streamlit 中显示
"""

import html as _html
import json

from pyvis.network import Network

# ---------- 常量：难度配色 ----------
DIFFICULTY_COLORS = {
    "基础": "#2ecc71",   # 绿色
    "中等": "#f39c12",   # 橙色
    "困难": "#e74c3c",   # 红色
}
DEFAULT_COLOR = "#95a5a6"   # 未知难度兜底灰色
EDGE_COLOR = "#e74c3c"      # 先修关系边统一红色

# ---------- 节点点击 -> 右侧详情面板 JS 模板 ----------
# 依赖 pyvis 生成 HTML 中的全局变量 network（vis.Network 实例）。
# 面板用 position:fixed 靠右侧停靠（top/right/bottom 归零），在 Streamlit iframe
# 内部闭环显示，不与 Streamlit 主页面产生任何通信（无需自定义组件）。
# 展示内容：知识点名称、难度徽章、简介、依赖关系（先修基础/后续可学）、课件原文出处。
OVERLAY_JS_TEMPLATE = """
<script>
(function () {
  var SOURCES = __SOURCES_JSON__;
  var DIFF_COLORS = {"基础": "#2ecc71", "中等": "#f39c12", "困难": "#e74c3c"};
  // 详情面板容器：固定停靠在画布右侧，内容超长时面板内部滚动
  var panel = document.createElement('div');
  panel.style.cssText = 'display:none;position:fixed;top:0;right:0;bottom:0;width:36%;min-width:280px;'
    + 'z-index:9999;overflow:auto;background:#ffffff;border-left:1px solid #d2d2d7;'
    + 'box-shadow:-8px 0 30px rgba(0,0,0,.10);padding:18px 18px 28px;'
    + 'font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:13px;line-height:1.8;';
  document.body.appendChild(panel);

  // 小工具：把知识点名渲染成灰底小标签（依赖关系列表用）
  function tag(n) {
    return '<span style="background:#f5f5f7;border:1px solid #e8e8ed;border-radius:6px;'
         + 'padding:0 8px;margin:2px 4px 2px 0;display:inline-block;color:#1d1d1f;">' + n + '</span>';
  }

  // vis.js 点击事件：点节点打开右侧详情面板，点空白处收起
  network.on("click", function (params) {
    if (params.nodes && params.nodes.length > 0) {
      var name = String(params.nodes[0]);
      var info = SOURCES[name] || {};
      var html = '<button style="position:absolute;top:12px;right:14px;border:none;'
               + 'background:none;cursor:pointer;font-size:18px;color:#86868b;" '
               + 'onclick="this.parentNode.style.display=\\'none\\'">&times;</button>'
               + '<div style="font-weight:700;font-size:16px;margin-bottom:4px;padding-right:24px;">' + name + '</div>';
      // 难度徽章（配色与节点一致）
      var diff = info.difficulty || '未知';
      var dc = DIFF_COLORS[diff] || '#95a5a6';
      html += '<span style="display:inline-block;background:' + dc + ';color:#fff;'
           + 'border-radius:10px;padding:1px 12px;font-size:12px;margin:2px 0 10px;">难度：' + diff + '</span>';
      if (info.desc) {
        html += '<div style="color:#1d1d1f;margin-bottom:12px;">' + info.desc + '</div>';
      }
      // 依赖关系：前驱 = 先修基础（先学谁），后继 = 后续可学（学完它能学谁）
      var pre = info.prereq || [], nxt = info.next || [];
      if (pre.length || nxt.length) {
        html += '<div style="font-weight:600;margin-bottom:6px;">依赖关系（共 '
              + (pre.length + nxt.length) + ' 条）</div>'
              + '<div style="margin-bottom:6px;"><b>先修基础</b>：'
              + (pre.length ? pre.map(tag).join('') : '<span style="color:#86868b;">无，可从本知识点学起</span>')
              + '</div>'
              + '<div style="margin-bottom:12px;"><b>后续可学</b>：'
              + (nxt.length ? nxt.map(tag).join('') : '<span style="color:#86868b;">无后续依赖</span>')
              + '</div>';
      }
      // 课件原文出处（可溯源）
      if (info.text) {
        if (info.pages && info.pages.length) {
          html += '<div style="color:#86868b;font-size:12px;margin-bottom:4px;">出处：课件第 '
                + info.pages.join('、') + ' 页</div>';
        } else {
          html += '<div style="color:#86868b;font-size:12px;margin-bottom:4px;">出处：课件原文片段</div>';
        }
        html += '<div style="color:#1d1d1f;background:#f5f5f7;border-radius:8px;padding:8px 12px;">' + info.text + '</div>';
      } else {
        html += '<div style="color:#86868b;">未在课件原文中定位到该知识点的段落。</div>';
      }
      panel.innerHTML = html;
      panel.style.display = 'block';
    } else {
      panel.style.display = 'none';   // 点击空白处收起详情面板
    }
  });
})();
</script>
"""


def build_graph_html(graph, height="600px", physics=True, node_sources=None):
    """
    networkx 图 -> pyvis 交互式可视化 HTML 字符串。
    :param graph:   networkx.DiGraph。节点属性：difficulty（难度）、description（描述）；
                    边属性：type（关系类型，默认"先修"），语义 u -[先修]-> v 表示先学 u
    :param height:  画布高度（CSS 单位，如 "600px"）
    :param physics: 是否启用力导向物理布局（节点自动散开、可拖拽）
    :param node_sources: 知识点 -> 详情映射。字段：
                    {"pages": [页码], "text": "原文",          # 原文出处（可溯源）
                     "difficulty": "基础/中等/困难", "desc": "简介",  # 详情面板头部
                     "prereq": [前驱], "next": [后继]}          # 依赖关系（先修基础/后续可学）
                    传入后点击节点在右侧详情面板展示以上信息
    :return: HTML 字符串；图谱为空/为 None 时返回占位提示 HTML
    """
    if graph is None or graph.number_of_nodes() == 0:
        return "<p style='color:#888;font-family:sans-serif'>知识图谱为空，暂无可视化内容。</p>"

    # directed=True：先修关系有方向（箭头）；notebook=False：生成完整独立 HTML
    net = Network(height=height, directed=True, bgcolor="#ffffff",
                  font_color="#333333", notebook=False)

    # ---- 第 1 步：添加节点（颜色=难度，大小=度数，悬浮=描述） ----
    for name, attrs in graph.nodes(data=True):
        name = str(name)
        diff = str(attrs.get("difficulty") or "")
        desc = str(attrs.get("description") or "")
        net.add_node(
            name,
            label=name,                                        # 画布上显示的名称
            # 悬浮提示（支持换行）：内容源自用户课件 -> html.escape 防 XSS 注入
            title=_html.escape(f"{name}\n难度：{diff or '未知'}\n{desc}"),
            color=DIFFICULTY_COLORS.get(diff, DEFAULT_COLOR),
            size=min(12 + graph.degree(name) * 4, 40),         # 度数越高越大，封顶 40
        )

    # ---- 第 2 步：添加先修关系边（箭头 u -> v，悬浮显示学习顺序） ----
    for u, v, attrs in graph.edges(data=True):
        dep_type = str(attrs.get("type") or "先修")
        net.add_edge(
            str(u), str(v),
            label=dep_type,                                    # 边上显示关系类型
            title=_html.escape(f"先修关系：{u} -[{dep_type}]-> {v}（先学 {u}，再学 {v}）"),
            color=EDGE_COLOR,
            arrows="to",                                       # 箭头指向后学的知识点
        )

    # ---- 第 3 步：力导向布局参数（负 gravity 让节点彼此推开，避免重叠） ----
    if physics:
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)

    # pyvis >= 0.3.0：generate_html() 直接返回 HTML 字符串，无需写临时文件
    html_str = net.generate_html()

    # 字符集保障：确保 <head> 里有 UTF-8 声明（导出下载后双击打开中文不乱码）
    if "charset" not in html_str[:800].lower():
        html_str = html_str.replace("<head>", '<head><meta charset="utf-8">', 1)

    # ---- 第 4 步：注入"节点点击 -> 原文出处"详情框（纯前端 JS overlay，零新依赖） ----
    if node_sources:
        # 防注入：节点名与原文都做 HTML 转义（innerHTML 渲染时可还原显示）；
        # JSON 中的 "</" 转义，防止原文内容提前闭合 <script> 标签
        safe_sources = {}
        for name, info in node_sources.items():
            safe_sources[_html.escape(str(name))] = {
                "pages": [int(p) for p in (info.get("pages") or [])
                          if str(p).strip().isdigit()],
                "text": _html.escape(str(info.get("text") or "")),
                "difficulty": _html.escape(str(info.get("difficulty") or "")),
                "desc": _html.escape(str(info.get("desc") or "")),
                "prereq": [_html.escape(str(n)) for n in (info.get("prereq") or [])],
                "next": [_html.escape(str(n)) for n in (info.get("next") or [])],
            }
        src_json = json.dumps(safe_sources, ensure_ascii=False).replace("</", "<\\/")
        overlay_js = OVERLAY_JS_TEMPLATE.replace("__SOURCES_JSON__", src_json)
        if "</body>" in html_str:
            html_str = html_str.replace("</body>", overlay_js + "\n</body>")
        else:
            html_str += overlay_js
    return html_str
