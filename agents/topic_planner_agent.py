# -*- coding: utf-8 -*-
"""
TopicPlannerAgent —— AI 自主规划智能体（无课件主题规划）
=======================================================
职责：用户只输入一个学习主题（不上传课件），一次性完成——
  1. 调 DeepSeek 生成该主题的系统学习大纲：8-12 个核心知识点
     （名称 + 简介 + 难度）与它们之间的先修依赖关系（单次 API 调用）；
  2. 组装 networkx.DiGraph 知识图谱（与 OntologyAgent 相同的边语义：
     from -[先修]-> to，即先学 from 再学 to；节点属性 description/difficulty）；
  3. 对图谱做分层拓扑排序，生成一条"由浅入深"的有序学习路径
     （复用 PathAgent 的拓扑排序思想：nx.topological_sort + 环退化 DFS）。

与 OntologyAgent 的关系：OntologyAgent 需要先有课件文本块（AI 从原文抽取
知识点 + 两轮调用分析依赖）；本智能体无课件，主题即输入，知识点由 AI 直接
生成，因此只复用其"图组装约定"（节点/边属性结构），不再重复调 API。

返回与入库约定（app.page_topic_planner 消费）：
    plan()      -> {"knowledge_points": [...], "dependencies": [...]} 或 None（失败）
    normalize() -> 规范化后的 (kps, deps)：名称去重清洗、依赖端点过滤（非法边丢弃）
    build_graph()-> networkx.DiGraph
    plan_path() -> [{"order": 1, "stage": 1, "knowledge_point": "...", "reason": "..."}]
"""

import json
import logging

import networkx as nx

from agents.base_agent import BaseAgent


class TopicPlannerAgent(BaseAgent):
    """AI 自主规划智能体：主题 -> 学习大纲 -> 知识图谱 -> 学习路径"""

    def plan(self, topic):
        """
        生成主题学习大纲（单次 DeepSeek 调用）。
        :param topic: 用户输入的学习主题（如 "Transformer架构"）
        :return: {"knowledge_points": [...], "dependencies": [...]} 原始 AI JSON；
                 失败返回 None（last_error 已写入分类信息，调用方走 ai_fail_hint）
        """
        prompt = f"""你是一个教育专家。请为学习主题「{topic}」构建一个系统性的学习大纲：
提取 8-12 个核心知识点，并给出它们之间的先修依赖关系（学习顺序依赖：先学 A 才能学 B，记为 A -> B）。

严格要求：
1. knowledge_points 为对象数组，name 是简洁的知识点名称（8 字以内），description 是不超过 30 字的学习向简介，
   difficulty 只能是"基础"、"中等"、"困难"三选一（按掌握该知识点的难度判断）；
2. dependencies 中 from 和 to 必须是 knowledge_points 里出现过的名称，且 from != to；
   type 只能是"先修"；只标注确实存在学习依赖的关系，不硬造；
3. 知识点覆盖该主题的完整学习路径（从入门到进阶），由浅入深；
4. 只输出 JSON，不要输出任何其他文字，格式如下：
{{"knowledge_points": [{{"name": "知识点", "description": "简介", "difficulty": "基础"}}],
"dependencies": [{{"from": "知识点A", "to": "知识点B", "type": "先修"}}]}}"""
        return self.chat_json(prompt, temperature=0.4)

    # ---------- 结果规范化 ----------
    @staticmethod
    def normalize(data, max_nodes=12):
        """
        校验并规范化 AI 返回的大纲（模型输出不可信，逐项清洗）：
          - knowledge_points：兼容字符串数组 / 对象数组；名称去空白、去重、截断超长；
            数量裁剪到 3..max_nodes（少于 3 个不足以构成学习路径）；
          - dependencies：只保留两端都在知识点集合内的边（AI 幻觉端点直接丢弃），
            去自环、去重；缺 type 时补"先修"。
        :return: (kps, deps) 元组；结构非法时 kps 为空列表（调用方判空报错）
        :raises ValueError: data 不是 dict 或缺少 knowledge_points 键
        """
        if not isinstance(data, dict):
            raise ValueError("AI 返回内容不是 JSON 对象")
        raw_kps = data.get("knowledge_points")
        if not isinstance(raw_kps, list):
            raise ValueError("AI 返回缺少 knowledge_points")

        # ---- 知识点清洗：字符串/对象统一为 {"name", "description", "difficulty"} ----
        kps, seen = [], set()
        for item in raw_kps:
            if isinstance(item, str):
                item = {"name": item, "description": "", "difficulty": "中等"}
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()[:30]
            if not name or name in seen:
                continue
            seen.add(name)
            difficulty = str(item.get("difficulty", "中等")).strip()
            kps.append({
                "name": name,
                "description": str(item.get("description", "")).strip()[:60],
                # 难度白名单校验（OntologyAgent 同款三档），非法值回落"中等"
                "difficulty": difficulty if difficulty in ("基础", "中等", "困难") else "中等",
            })
            if len(kps) >= max_nodes:
                break

        # ---- 依赖清洗：端点必须都在知识点集合内 ----
        names = {k["name"] for k in kps}
        deps, dep_seen = [], set()
        for item in (data.get("dependencies") or []):
            if not isinstance(item, dict):
                continue
            u, v = str(item.get("from", "")).strip(), str(item.get("to", "")).strip()
            if u not in names or v not in names or u == v:
                continue
            key = (u, v)
            if key in dep_seen:
                continue
            dep_seen.add(key)
            deps.append({"from": u, "to": v, "type": "先修"})
        return kps, deps

    # ---------- 知识图谱组装（与 OntologyAgent.build 的节点/边结构一致） ----------
    @staticmethod
    def build_graph(kps, deps):
        """
        由规范化后的知识点与依赖组装知识图谱。
        边语义与 OntologyAgent 完全一致：from -[先修]-> to（先学 from 再学 to）。
        :return: networkx.DiGraph
        """
        graph = nx.DiGraph()
        for k in kps:
            graph.add_node(k["name"], description=k["description"],
                           difficulty=k["difficulty"])
        for d in deps:
            graph.add_edge(d["from"], d["to"], type=d["type"])
        return graph

    # ---------- 学习路径（分层拓扑排序，复用 PathAgent 的拓扑思想） ----------
    @staticmethod
    def plan_path(graph):
        """
        对整张图谱做分层拓扑排序，生成"由浅入深"的学习路径。
        与 PathAgent.generate 的差异：PathAgent 以"诊断薄弱点"为驱动（先修祖先
        闭包子图排序）；本场景没有诊断报告，直接对全图排序，并按 Kahn 入度分层——
        同一层（可并行学习）的知识点归入同一 stage，让路径呈现清晰的进阶节奏。
        环处理与 PathAgent 一致：拓扑排序失败时退化为 DFS 逆后序近似。
        :return: [{"order": 1, "stage": 1, "knowledge_point": "...", "reason": "..."}]
        """
        if graph is None or graph.number_of_nodes() == 0:
            return []

        # ---- Kahn 分层：每轮取走当前入度为 0 的节点（= 本层可学，先修已全部完成） ----
        remaining = set(graph.nodes)
        in_deg = dict(graph.in_degree())
        frontier = [n for n in graph.nodes if in_deg[n] == 0]
        layers = []
        while frontier:
            layers.append(sorted(frontier))   # 同层按名称排序，输出稳定
            nxt = []
            for node in frontier:
                for succ in graph.successors(node):
                    in_deg[succ] -= 1
                    if in_deg[succ] == 0:
                        nxt.append(succ)
            remaining -= set(frontier)
            frontier = nxt
        if remaining:   # 图有环：剩余节点无法分层，按名称追加在末尾（保证全覆盖）
            layers.append(sorted(remaining))

        # ---- 生成带理由的有序路径 ----
        path, order = [], 0
        for stage, layer in enumerate(layers, start=1):
            for name in layer:
                order += 1
                succs = [str(s) for s in graph.successors(name)]
                preds = [str(p) for p in graph.predecessors(name)]
                if not preds:
                    reason = "本主题的入门起点，无先修要求，建议首先掌握"
                elif succs:
                    reason = f"掌握后可解锁「{'、'.join(succs[:2])}」等后续内容"
                else:
                    reason = "进阶收尾内容，建立在前面所有知识点之上"
                desc = (graph.nodes[name] or {}).get("description") or ""
                if desc:
                    reason = f"{desc}（{reason}）"
                path.append({"order": order, "stage": stage,
                             "knowledge_point": name, "reason": reason})
        return path


# 轻量 logger（与 PathAgent 同理：本类主要走 BaseAgent 的日志体系，这里仅备查）
logger = logging.getLogger("TopicPlannerAgent")
