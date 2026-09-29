# -*- coding: utf-8 -*-
"""
TutorAgent —— 答疑智能体（引导式辅导，AI 内容强制绑定来源）
==========================================================
职责：基于检索到的课件片段与知识点回答学生问题。
教学风格：先引导学生思考、不直接给答案；回答内容必须来自课件。

强制来源绑定：调用大模型时强制返回 JSON——
    {"content": "回答正文", "source_page": 依据页码, "source_snippet": "引用原文片段"}
模型未按要求返回 JSON 时，自动降级为纯文本回答（来源字段留空），保证答疑永不中断。

输入：
    question         用户问题
    contexts         检索到的片段 [{"id", "text", "score", "pages"}]（RetrieverAgent 的
                     chunks；pages 为 PDF 页码列表，供模型标注来源页码）
    knowledge_points 图谱命中的知识点 [{"name", "description", "difficulty",
                      "match_type"}]（RetrieverAgent 的 knowledge_points，可选）

输出：
    {"content": str, "source_page": int 或 None, "source_snippet": str 或 None}
    API 调用失败返回 None，由调用方（app.py）提示错误。

API Key：继承 BaseAgent —— 自动从项目根目录 .env 读取 DEEPSEEK_API_KEY。
"""

from .base_agent import BaseAgent

# ---------- 系统提示词：约束教学风格 + 强制 JSON 输出（AI 内容绑定来源） ----------
SYSTEM_PROMPT = """你是一个学习辅导助手。请根据以下课程内容回答问题。
要求：
1. 先引导学生思考，不要直接给答案
2. 回答简洁，内容必须来自课程内容，不得编造
3. 如果涉及知识点，指出相关知识点名称
4. 只输出一个 JSON 对象，不要输出任何解释文字或代码块标记"""

# 降级路径的系统提示词（模型未按 JSON 返回时回退纯文本，仅保留教学风格约束）
FALLBACK_SYSTEM_PROMPT = """你是一个学习辅导助手。请根据以下课程内容回答问题。
要求：1. 先引导学生思考，不要直接给答案 2. 回答简洁，内容必须来自课程内容"""


class TutorAgent(BaseAgent):
    """答疑智能体：基于课件片段做引导式辅导，片段中没有的信息不编造"""

    def ask(self, question, contexts, knowledge_points=None, history=None):
        """
        回答学生问题（强制 JSON，AI 内容绑定来源）。
        :param question: 用户问题
        :param contexts: 检索片段 [{"id", "text", "score", "pages"}]
        :param knowledge_points: 图谱命中的知识点列表（可选；无图谱时传 None）
        :param history: 多轮对话历史 [{"role": "user"/"assistant", "content": str}, ...]
                        （可选；传入后模型能理解"没听懂，再讲讲"这类追问）
        :return: {"content": 回答正文(Markdown),
                  "source_page": 回答依据的页码(int)或 None,
                  "source_snippet": 引用的课件原文片段(str)或 None}
                 API 失败返回 None
        """
        # ---- 第 1 步：拼接课程内容上下文（片段带页码标注，供模型引用来源页码） ----
        ctx_lines = []
        for c in contexts:
            pages = c.get("pages") or []
            # 跨页块显示起始页即可（来源标注只需一个主要页码）
            tag = f"｜第{pages[0]}页" if pages else ""
            ctx_lines.append(f"【{c['id']}{tag}】\n{c['text']}")
        context = "\n\n".join(ctx_lines)

        # 知识点上下文（混合检索的第二路结果）：让模型能"指出相关知识点名称"
        if knowledge_points:
            kp_lines = "\n".join(
                f"- {p['name']}（难度：{p['difficulty']}，{p['match_type']}）"
                f"：{p['description'] or '暂无描述'}"
                for p in knowledge_points
            )
            context += f"\n\n相关知识点：\n{kp_lines}"

        # ---- 第 2 步：用户提示词 = 课程内容 + 历史对话 + 问题 + JSON 输出格式 ----
        # 历史对话取最近 5 轮即可：太多会挤占上下文、稀释课件内容的权重
        history_block = ""
        recent = (history or [])[-10:]   # 10 条消息 = 5 轮对话
        if recent:
            lines = [f"{'学生' if m['role'] == 'user' else '助手'}：{m['content']}" for m in recent]
            history_block = f"历史对话（供理解上下文，不必重复回答）：\n" + "\n".join(lines) + "\n\n"

        prompt = f"""课程内容：
{context}

{history_block}用户问题：{question}

输出格式（严格 JSON，只输出 JSON 本身，不要有任何其他文字）：
{{"content": "回答正文（Markdown）", "source_page": 回答主要依据的页码（整数，上下文未标注页码时填 null）, "source_snippet": "你的回答所依据的课程原文片段（原样摘录，60字以内）"}}
若用户问题是对前面对话的追问，请结合历史对话理解后再回答。"""

        # ---- 第 3 步：调用 DeepSeek，强制 JSON 并规范化来源字段 ----
        # temperature=0.3：内容必须严格来自课件，低温减少编造
        data = self.chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.3)
        if isinstance(data, dict) and str(data.get("content", "")).strip():
            try:
                page = int(data.get("source_page"))
            except (TypeError, ValueError):
                page = None   # 模型返回 null / 非数字 / 缺字段
            snippet = str(data.get("source_snippet") or "").strip() or None
            return {"content": str(data["content"]).strip(),
                    "source_page": page, "source_snippet": snippet}

        # ---- 第 4 步：降级——模型未按 JSON 返回时回退纯文本回答（来源字段留空） ----
        self.logger.warning("TutorAgent 未收到合法 JSON 回答，降级为纯文本输出")
        text = self.chat(prompt, system=FALLBACK_SYSTEM_PROMPT, temperature=0.3)
        if text is None:
            return None   # API 调用失败：last_error 已由 chat() 记录
        return {"content": text.strip(), "source_page": None, "source_snippet": None}
