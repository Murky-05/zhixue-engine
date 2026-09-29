# -*- coding: utf-8 -*-
"""
PathAgent —— 学习路径智能体（拓扑排序版）
==========================================
职责：根据诊断报告中的薄弱知识点 + 知识图谱先修依赖，用拓扑排序
     生成一条"先补前置、再攻薄弱"的有序学习路径。

纯本地算法：不需要 API Key（与 IndexAgent 同理），零 API 成本。

处理流程：
    1. 解析诊断报告，提取薄弱知识点（weak_points，支持 dict / JSON 字符串）
    2. 从图谱收集每个薄弱点的全部先修节点（nx.ancestors 祖先闭包，
       边语义 from -[先修]-> to，即先学 from 再学 to）
    3. 对「先修节点 + 薄弱点」构造子图做拓扑排序 —— 薄弱点的祖先
       天然排在它前面，薄弱点之间的先修顺序也同时被保证
    4. 图有环导致拓扑排序失败时，退化为 DFS 逆后序做近似拓扑排序
    5. 为每个节点生成学习理由 reason；图谱未收录的薄弱点排在末尾

返回格式：
    [{"order": 1, "knowledge_point": "叶绿体", "reason": "..."}, ...]
"""

import json
import logging

import networkx as nx


class PathAgent:
    """学习路径智能体：先修依赖拓扑排序（纯本地算法，不需要 API Key）"""

    def generate(self, report, graph=None):
        """
        生成有序学习路径。
        :param report: 诊断报告 dict（DiagnosisAgent.diagnose 的返回值）
                       或其 JSON 字符串（兼容旧调用）
        :param graph:  知识图谱 networkx.DiGraph（可选；为空时退化为薄弱点清单）
        :return: [{"order": 1, "knowledge_point": "...", "reason": "..."}]；
                 报告中无薄弱点返回 []
        """
        weak_info = self._extract_weak_points(report)
        if not weak_info:
            return []

        # ---- 无图谱兜底：按报告出现顺序给出薄弱点清单 ----
        if graph is None or graph.number_of_nodes() == 0:
            return [
                {"order": i, "knowledge_point": kp,
                 "reason": "诊断答错的知识点（暂无知识图谱依赖可参考，建议结合课件逐个复习）"}
                for i, kp in enumerate(weak_info, start=1)
            ]

        # ---- 第 1 步：收集每个薄弱点的全部先修节点（祖先闭包） ----
        in_graph = [kp for kp in weak_info if kp in graph]      # 图谱收录的薄弱点
        missing = [kp for kp in weak_info if kp not in graph]   # 图谱未收录的薄弱点
        prereq_set = set()
        for kp in in_graph:
            prereq_set |= nx.ancestors(graph, kp)   # 所有可达上游节点 = 全部先修链

        # ---- 第 2 步：构造子图（先修 + 薄弱点）并拓扑排序 ----
        # 把薄弱点一并放入子图排序：边语义保证所有先修自然排在薄弱点之前，
        # 等价于"先对先修节点排序，再把薄弱点插入其依赖之后"
        sub = graph.subgraph(prereq_set | set(in_graph))
        try:
            ordered = list(nx.topological_sort(sub))
        except nx.NetworkXUnfeasible:
            # ---- 图有环：用 DFS 逆后序近似 ----
            # DAG 中逆后序就是合法拓扑序；有环时它是稳定可用的近似序
            logger.warning("图谱存在循环依赖，使用 DFS 逆后序近似拓扑排序")
            ordered = list(nx.dfs_postorder_nodes(sub))[::-1]

        # ---- 第 3 步：生成带理由的有序路径 ----
        path = []
        for node in ordered:
            node = str(node)
            if node in weak_info:
                # 薄弱点：理由 = 答错 + 知识域 + 历史重复度
                info = weak_info[node]
                parts = ["诊断答错，需重点巩固"]
                if info["domain"]:
                    parts.append(f"知识域：{info['domain']}")
                if info["recurrence"]:
                    parts.append(f"历史已薄弱 {info['recurrence']} 次，建议专项复习")
                path.append({"knowledge_point": node, "reason": "；".join(parts)})
            else:
                # 先修节点：理由 = 它是谁的基础（取子图中的直接下游）
                dependents = [str(s) for s in sub.successors(node)][:2]
                target = "、".join(dependents) if dependents else "后续知识点"
                path.append({
                    "knowledge_point": node,
                    "reason": f"「{target}」的先修知识，先掌握它才能理解后续内容",
                })

        # ---- 第 4 步：图谱未收录的薄弱点排在末尾 ----
        for kp in missing:
            path.append({
                "knowledge_point": kp,
                "reason": "诊断答错的知识点（图谱中未收录，建议结合课件原文复习）",
            })

        return [{"order": i, **step} for i, step in enumerate(path, start=1)]

    # ---------- 诊断报告解析 ----------
    @staticmethod
    def _extract_weak_points(report):
        """
        从诊断报告提取薄弱知识点（保序去重）。
        :param report: dict 或 JSON 字符串
        :return: {知识点名: {"domain": str, "recurrence": int}}，无薄弱点返回 {}
        """
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except json.JSONDecodeError:
                return {}   # 旧版 Markdown 报告无法解析
        if not isinstance(report, dict):
            return {}

        weak = report.get("weak_points") or []
        info, seen = {}, set()
        for w in weak:
            if isinstance(w, dict):
                name = str(w.get("knowledge_point", "")).strip()
                domain = str(w.get("domain", "")).strip()
                recurrence = w.get("recurrence", 0)
            else:
                name, domain, recurrence = str(w).strip(), "", 0
            if name and name not in seen:
                seen.add(name)
                info[name] = {
                    "domain": domain,
                    "recurrence": recurrence if isinstance(recurrence, int) else 0,
                }
        return info


# 纯本地算法不继承 BaseAgent，独立使用轻量 logger
logger = logging.getLogger("PathAgent")
