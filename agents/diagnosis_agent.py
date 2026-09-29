# -*- coding: utf-8 -*-
"""
DiagnosisAgent —— 学习诊断智能体
================================
职责：从知识图谱挑选核心知识点出选择题，批改后生成结构化诊断报告（JSON）。

流程一：get_questions(graph, n, difficulty, chunks=None)
    1. 按节点度数（入度+出度）降序排列，度数越高说明与越多知识点存在
       先修关联，越接近"核心概念"，取前 n 个；
    2. 逐个调用 DeepSeek 为每个知识点生成 1 道单选题（严格 JSON）；
       传入 chunks 时会把知识点所在课件原文段落注入出题上下文，
       并要求题目标注 source_page（页码）与 source_snippet（原文片段）；
    3. 字段校验（题目/选项/答案合法性），无效题目跳过。

流程二：diagnose(answers, quiz, graph=None, history=None)
    1. 本地判分：逐题比对选项，不依赖 AI；
    2. 知识域：AI 标记答错知识点所属章节/知识域；
    3. 正确性：AI 分析错误类型（概念性 / 程序性 / 粗心）；
    4. 重复度：从 SQLite 历史诊断记录统计同一薄弱知识点的出现次数；
    5. 组装详细报告 JSON（逐题明细 + 薄弱点 + 错误类型 + 建议）。
       AI 分析失败时自动降级为本地规则建议，保证诊断永不中断。

报告格式：
    {
        "score": 2, "total": 3,
        "detail":      [{"index", "knowledge_point", "your_answer",
                         "correct_answer", "result", "explain",
                         "source_page", "source_snippet"}],
        "mastery":     [{"knowledge_point", "correct", "total",
                         "status": "掌握"/"薄弱"}],
        "weak_points": [{"knowledge_point", "domain", "recurrence"}],
        "error_types": [{"knowledge_point", "type", "analysis"}],
        "suggestions": ["...", "..."]
    }
"""

import json
from collections import Counter

from .base_agent import BaseAgent

# ---------- 常量 ----------
VALID_ERROR_TYPES = {"概念性", "程序性", "粗心"}   # 合法错误类型（超出范围归为概念性）
ERROR_TYPE_PROMPT = "概念性（概念理解错误）/ 程序性（方法步骤用错）/ 粗心（会做但看错算错）"

# 出题难度风格说明（注入出题提示词，控制题目风格）
DIFFICULTY_PROMPTS = {
    "基础": "基础难度：只考察概念辨认与定义理解，题干直白，选项区分明显",
    "进阶": "进阶难度：考察知识的应用与分析，需理解原理、辨析易混概念才能作答",
    "综合": "综合难度：混合概念辨析与情境应用，可结合前后置知识点设置综合情境",
}


class DiagnosisAgent(BaseAgent):
    """诊断智能体：图谱选点出题 -> 本地判分 -> AI 错因分析 -> 结构化报告"""

    # ================= 流程一：出题 =================
    def get_questions(self, graph, n=3, difficulty="综合", chunks=None):
        """
        从知识图谱中选取 n 个核心知识点，各生成 1 道选择题（AI 内容绑定来源）。
        :param graph: OntologyAgent 构建的知识图谱（networkx.DiGraph）
        :param n: 题目数量
        :param difficulty: 出题难度（基础/进阶/综合），控制题目风格
        :param chunks: 课件文本块列表（可选）。传入后把知识点所在原文段落注入出题
                       上下文，题目标注来源页码与原文片段（AI 生成内容可溯源）
        :return: 题目列表 [{"knowledge_point", "question", "options",
                 "answer", "explain", "source_page", "source_snippet"}]；
                 图谱为空或全部生成失败返回 []
        """
        if graph is None or graph.number_of_nodes() == 0:
            self.logger.warning("知识图谱为空，无法出题")
            return []

        # 第 1 步：按度数排序选核心知识点（度数 = 入度 + 出度）
        ranked = sorted(graph.degree, key=lambda x: (-x[1], str(x[0])))
        top_points = [str(name) for name, _ in ranked[:n]]
        self.logger.info("核心知识点（按度数排序）：%s（难度：%s）", top_points, difficulty)

        # 第 2 步：逐个知识点出题（单点失败只跳过该题，不影响其他题）
        questions = []
        for kp in top_points:
            chunk = self._find_chunk(kp, chunks)   # 定位知识点所在课件原文
            q = self._make_question(kp, graph, difficulty, chunk=chunk)
            if q:
                questions.append(q)
        return questions

    @staticmethod
    def _find_chunk(knowledge_point, chunks):
        """定位知识点首次出现的文本块（出题的原文依据）；未命中返回 None"""
        for c in chunks or []:
            if knowledge_point in (c.get("text") or ""):
                return c
        return None

    def _make_question(self, knowledge_point, graph, difficulty="综合", chunk=None):
        """
        为单个知识点生成 1 道单选题。
        提示词中注入图谱上下文（描述/难度/前后置知识）、课件原文段落与出题难度风格，
        让题目既贴合课件内容，又符合用户选择的难度；强制返回来源页码与原文片段。
        """
        attrs = graph.nodes[knowledge_point]
        prereq = [str(p) for p in graph.predecessors(knowledge_point)]   # 前置知识
        follow = [str(s) for s in graph.successors(knowledge_point)]     # 后续知识

        context = (
            f"知识点：{knowledge_point}\n"
            f"描述：{attrs.get('description', '')}\n"
            f"难度：{attrs.get('difficulty', '中等')}"
        )
        if prereq:
            context += f"\n前置知识：{'、'.join(prereq)}"
        if follow:
            context += f"\n后续知识：{'、'.join(follow)}"

        # 课件原文上下文：有原文时要求题目依据原文出题，并标注来源页码/片段；
        # 无原文，或原文无页码（docx/txt/md）时，页码填空值（不得编造页码）
        if chunk:
            pages = chunk.get("pages") or []
            page_tag = f"（第{pages[0]}页）" if pages else ""
            context += f"\n课件原文【{chunk['id']}{page_tag}】：\n{chunk['text']}"
            if pages:
                source_rule = "source_page 填题目依据的页码（整数），source_snippet 填出题依据的原文片段（原样摘录，60字以内）"
                example = '"source_page": 3, "source_snippet": "原文片段"'
            else:
                source_rule = "课件原文未标注页码，source_page 填 null（不得编造页码）；source_snippet 填出题依据的原文片段（原样摘录，60字以内）"
                example = '"source_page": null, "source_snippet": "原文片段"'
        else:
            source_rule = "source_page 填 null，source_snippet 填空字符串"
            example = '"source_page": null, "source_snippet": ""'

        style = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["综合"])
        prompt = f"""请根据以下知识点信息，出 1 道考察该知识点的单项选择题。

{context}

题目风格：{style}

严格要求：
1. question 为题干；options 为 4 个选项（键为 A/B/C/D）；answer 为正确选项字母；
   explain 为一句话解析（说明为什么正确答案对）；{source_rule}；
2. 干扰项要有迷惑性但必须唯一正确；
3. 只输出 JSON，不要输出任何其他文字：
{{"question": "题干", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "B", "explain": "解析", {example}}}"""
        data = self.chat_json(prompt, temperature=0.5)   # 出题允许适度发散
        return self._validate_question(data, knowledge_point)

    @staticmethod
    def _validate_question(data, knowledge_point):
        """校验并清洗单题 JSON：选项键规范为 A-D、答案必须在选项中，不合格返回 None"""
        if not isinstance(data, dict):
            return None
        question = str(data.get("question", "")).strip()
        raw_options = data.get("options")
        answer = str(data.get("answer", "")).strip().upper()[:1]
        if not question or not isinstance(raw_options, dict) or len(raw_options) < 2:
            return None
        # 选项键统一为大写单字母，值转字符串，最多取 4 个
        options = {str(k).strip().upper()[:1]: str(v).strip()
                   for k, v in list(raw_options.items())[:4]}
        if answer not in options:
            return None   # 答案不在选项里 => 题目无效
        # 来源字段规范化：页码转 int（失败置 None），片段截断 80 字防过长
        try:
            page = int(data.get("source_page"))
        except (TypeError, ValueError):
            page = None
        snippet = str(data.get("source_snippet") or "").strip()[:80] or None
        return {
            "knowledge_point": str(knowledge_point),
            "question": question,
            "options": options,
            "answer": answer,
            "explain": str(data.get("explain", "")).strip(),
            "source_page": page,
            "source_snippet": snippet,
        }

    # ================= 流程二：诊断 =================
    def diagnose(self, answers, quiz, graph=None, history=None):
        """
        批改作答结果并生成诊断报告。
        :param answers: 学生答案列表（与 quiz 等长对齐，元素为选项字母，可为空串）
        :param quiz: get_questions 返回的题目列表
        :param graph: 知识图谱（可选，用于生成"先补前置知识"类建议）
        :param history: SQLite 历史记录列表（app.py 传入 utils.db.get_history() 的结果，
                        从中筛选 diagnosis 类型解析历史薄弱知识点）
        :return: 诊断报告 dict（格式见模块 docstring）；quiz 为空返回 None
        """
        if not quiz:
            return None

        # ---- 1) 本地判分（逐题比对，不依赖 AI） ----
        detail, wrongs = [], []
        for i, (q, a) in enumerate(zip(quiz, answers)):
            yours = str(a or "").strip().upper()[:1]
            correct = str(q.get("answer", "")).strip().upper()[:1]
            ok = yours == correct and yours != ""
            detail.append({
                "index": i + 1,
                "knowledge_point": q.get("knowledge_point", ""),
                "your_answer": yours,
                "correct_answer": correct,
                "result": "正确" if ok else "错误",
                "explain": str(q.get("explain", "")).strip(),   # 题目解析（错题会被 AI 分析覆盖）
                # 题目来源（AI 生成内容可溯源）：出题依据的页码与原文片段
                "source_page": q.get("source_page"),
                "source_snippet": q.get("source_snippet"),
            })
            if not ok:
                wrongs.append({"index": i + 1, "q": q, "yours": yours})
        score = sum(1 for d in detail if d["result"] == "正确")
        total = len(detail)

        # ---- 2) 重复度：历史诊断报告中同一薄弱知识点的出现次数 ----
        hist_counts = self._count_history_weak_points(history)

        # ---- 3) AI 错因分析（一次调用：知识域 + 错误类型 + 建议） ----
        analysis = self._analyze_wrongs(wrongs) if wrongs else {"items": [], "suggestions": []}

        # ---- 4) 组装薄弱知识点与错误类型（答错知识点去重合并） ----
        weak_points, error_types = [], []
        seen_kp = set()
        for w in wrongs:
            kp = w["q"].get("knowledge_point", "")
            item = next((it for it in analysis["items"] if it["knowledge_point"] == kp), None)
            if kp not in seen_kp:
                seen_kp.add(kp)
                weak_points.append({
                    "knowledge_point": kp,
                    "domain": (item or {}).get("domain", ""),            # 所属章节/知识域
                    "recurrence": hist_counts.get(kp, 0),                # 历史出现次数
                })
            error_types.append({
                "knowledge_point": kp,
                "type": (item or {}).get("error_type", "概念性"),
                "analysis": (item or {}).get("analysis", w["q"].get("explain", "")),
            })
            # 错题解析回填到逐题明细（AI 错因分析优先，题目自带解析兜底）
            detail[w["index"] - 1]["explain"] = (item or {}).get("analysis") or w["q"].get("explain", "")

        # ---- 4.5) 知识点掌握度：按知识点聚合对错（同一知识点多题合并统计） ----
        agg = {}
        for d in detail:
            c, t = agg.get(d["knowledge_point"], (0, 0))
            agg[d["knowledge_point"]] = (c + (1 if d["result"] == "正确" else 0), t + 1)
        mastery = [
            {
                "knowledge_point": kp,
                "correct": c,
                "total": t,
                "status": "掌握" if c == t else "薄弱",
            }
            for kp, (c, t) in agg.items()
        ]
        mastery.sort(key=lambda m: (m["status"] != "薄弱", -m["total"]))   # 薄弱点排前

        # ---- 5) 学习建议：AI 建议 + 图谱前置提示 + 本地兜底 ----
        suggestions = list(analysis["suggestions"])
        for kp in seen_kp:
            hint = self._prereq_hint(kp, graph)
            if hint:
                suggestions.append(hint)
        if not suggestions:
            # AI 分析失败 / 无建议时的本地兜底规则
            suggestions = [f"重新复习「{kp}」对应的课件片段，并完成同类练习" for kp in seen_kp]
            suggestions.append("把错题讲给别人听（费曼学习法），检验是否真正理解")

        return {
            "score": score,
            "total": total,
            "detail": detail,
            "mastery": mastery,
            "weak_points": weak_points,
            "error_types": error_types,
            "suggestions": suggestions[:6],   # 建议最多 6 条，避免冗长
        }

    # ---------- AI 错因分析 ----------
    def _analyze_wrongs(self, wrongs):
        """
        一次 DeepSeek 调用完成三件事：标记知识域（章节）、分析错误类型、给建议。
        :return: {"items": [{"knowledge_point", "domain", "error_type", "analysis"}],
                  "suggestions": [str]}；调用失败返回空结构（由调用方兜底）
        """
        wrong_lines = "\n".join(
            f"- 第{w['index']}题（知识点「{w['q']['knowledge_point']}」）：{w['q']['question']}\n"
            f"  学生选 {w['yours'] or '未作答'}，正确答案 {w['q']['answer']}。题目解析：{w['q'].get('explain', '')}"
            for w in wrongs
        )
        prompt = f"""以下是学生答错的选择题，请逐题分析。

{wrong_lines}

严格要求：
1. items 中每道错题一项：knowledge_point 照抄原知识点名称；
   domain 推断该知识点所属的章节/知识域（如"第二章 细胞结构"）；
   error_type 只能是：{ERROR_TYPE_PROMPT}；
   analysis 为 50 字内的错因分析（结合学生的错误选项推断思路偏差）；
2. suggestions 为 2-4 条针对这些薄弱点的具体学习建议；
3. 只输出 JSON：
{{"items": [{{"knowledge_point": "...", "domain": "...", "error_type": "概念性", "analysis": "..."}}],
  "suggestions": ["建议1", "建议2"]}}"""

        data = self.chat_json(prompt, temperature=0.3)   # 诊断分析要严谨
        return self._validate_analysis(data)

    @staticmethod
    def _validate_analysis(data):
        """校验 AI 分析结果：错误类型白名单过滤，字段缺失补默认值，失败返回空结构"""
        empty = {"items": [], "suggestions": []}
        if not isinstance(data, dict):
            return empty
        items = []
        raw_items = data.get("items")
        for it in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(it, dict):
                continue
            kp = str(it.get("knowledge_point", "")).strip()
            if not kp:
                continue
            error_type = str(it.get("error_type", "")).strip()
            items.append({
                "knowledge_point": kp,
                "domain": str(it.get("domain", "")).strip()[:50],
                "error_type": error_type if error_type in VALID_ERROR_TYPES else "概念性",
                "analysis": str(it.get("analysis", "")).strip()[:200],
            })
        sug_raw = data.get("suggestions")
        suggestions = [str(s).strip()[:100] for s in sug_raw
                       if str(s).strip()] if isinstance(sug_raw, list) else []
        return {"items": items, "suggestions": suggestions}

    # ---------- 历史重复度统计 ----------
    @staticmethod
    def _count_history_weak_points(history):
        """
        从历史记录中统计各知识点作为"薄弱点"出现的次数。
        :param history: 历史记录列表（app.py 传入 utils.db.get_history() 的结果；
                        也兼容旧版单表记录格式）
        :return: Counter {知识点名: 出现次数}
        兼容性：
          - 新格式 get_history(): {"type": "diagnosis", "detail": {"report": "<JSON>"}}
          - 旧格式 records 表:    {"record_type": "diagnosis", "answer": "<JSON>"}
          - 解析失败（如 Markdown 文本）的记录直接跳过。
        """
        counts = Counter()
        for rec in history or []:
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type") or rec.get("record_type")
            if rtype != "diagnosis":
                continue
            # 报告 JSON 文本：新格式在 detail.report 里，旧格式在 answer 字段
            text = rec.get("answer")
            if text is None:
                text = (rec.get("detail") or {}).get("report")
            try:
                data = json.loads(text or "")
            except (json.JSONDecodeError, TypeError):
                continue   # 旧版 Markdown 记录，无法解析，跳过
            weak = data.get("weak_points") if isinstance(data, dict) else None
            for w in weak if isinstance(weak, list) else []:
                name = str(w.get("knowledge_point", "")).strip() if isinstance(w, dict) else str(w).strip()
                if name:
                    counts[name] += 1
        return counts

    # ---------- 图谱前置提示 ----------
    @staticmethod
    def _prereq_hint(knowledge_point, graph):
        """薄弱知识点存在前置知识时，生成"先补前置"的建议；无图谱返回 None"""
        if graph is None or knowledge_point not in graph:
            return None
        prereq = [str(p) for p in graph.predecessors(knowledge_point)]
        if not prereq:
            return None
        return f"「{knowledge_point}」依赖前置知识：{'、'.join(prereq)}，建议先巩固前置内容再回炉本点"
