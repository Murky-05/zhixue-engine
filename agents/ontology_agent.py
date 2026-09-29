# -*- coding: utf-8 -*-
"""
OntologyAgent —— 知识图谱构建智能体
===================================
职责：把 ParserAgent 产出的「知识点候选列表」加工成一张知识图谱。

处理流程（两次 DeepSeek 调用）：
    1. 知识点属性补全：为每个知识点生成名称、描述、难度（基础/中等/困难）
    2. 先修关系分析：分析知识点之间的学习依赖（先学谁才能学谁）
    3. 用 networkx.DiGraph 组装：节点=知识点，边=先修关系

返回：networkx.DiGraph 对象
    - 节点属性：description（描述）、difficulty（难度）
    - 边属性：type（"先修"）；语义为 from -> to，即先学 from 才能学 to

使用示例：
    from agents.ontology_agent import OntologyAgent
    graph = OntologyAgent().build(["牛顿第二定律", "加速度"], chunks=文本块列表)
    graph.nodes(data=True)   # 查看节点及属性
    graph.edges(data=True)   # 查看边及属性
"""

import json

import networkx as nx

from .base_agent import BaseAgent

# 合法取值（与提示词保持一致，用于校验清洗）
VALID_DIFFICULTIES = {"基础", "中等", "困难"}   # 难度三档
VALID_TYPES = {"先修"}                          # 关系类型（当前仅先修）


class OntologyAgent(BaseAgent):
    """知识图谱构建智能体：知识点候选列表 -> networkx 有向图"""

    def build(self, knowledge_candidates, chunks=None, max_nodes=25):
        """
        构建知识图谱主入口。
        :param knowledge_candidates: 知识点候选名称列表（来自 ParserAgent）
        :param chunks: 文本块列表 [{"id", "text"}]（可选，为难度与依赖判断提供课件上下文）
        :param max_nodes: 最多纳入图谱的知识点数量（控制 API 长度与图谱可读性）
        :return: networkx.DiGraph；节点补全失败时返回空图（不抛异常）
        """
        # 预处理：候选名称去重、限量（保持出现顺序）
        candidates, seen = [], set()
        for k in (knowledge_candidates or [])[:max_nodes]:
            k = str(k).strip()
            if k and k not in seen:
                seen.add(k)
                candidates.append(k)
        if not candidates:
            self.logger.warning("没有知识点候选，跳过图谱构建")
            return nx.DiGraph()

        # 第 1 步：调用 DeepSeek 补全知识点属性（名称/描述/难度）
        nodes = self._enrich_nodes(candidates, chunks)
        if not nodes:
            self.logger.warning("知识点属性补全失败，返回空图谱")
            return nx.DiGraph()

        # 第 2 步：调用 DeepSeek 分析知识点之间的先修关系
        deps = self._analyze_dependencies(nodes, chunks)

        # 第 3 步：用 networkx 组装有向图（节点=知识点，边=先修关系）
        graph = nx.DiGraph()
        for node in nodes:
            graph.add_node(
                node["name"],
                description=node["description"],
                difficulty=node["difficulty"],
            )
        for dep in deps:
            graph.add_edge(dep["from"], dep["to"], type=dep["type"])
        self.logger.info(
            "图谱构建完成：%d 个节点，%d 条先修关系", graph.number_of_nodes(), graph.number_of_edges()
        )
        return graph

    # ---------- 增量构建：只分析新增知识点（旧节点/旧边复用，省 API 成本） ----------
    def merge_build(self, knowledge_candidates, chunks=None, existing_graph=None, max_nodes=25):
        """
        在已有图谱基础上增量并入新知识点：
          - 已存在于图谱中的候选 -> 直接复用（不调 AI）；
          - 新候选 -> 仅对它们补全属性（1 次 AI 调用）；
          - 先修关系 -> 仅分析"至少一端是新知识点"的关系（1 次 AI 调用），
            旧知识点之间的既有关系原样保留，不做全量重分析。
        :param existing_graph: 已有的 networkx.DiGraph（为空/None 时退化为全量 build）
        :return: (graph, added_count) 二元组：合并后的图 + 实际新增的节点数
        """
        # 无已有图谱：没有可复用的基础，退化为全量构建（新增数 = 全部节点数）
        if existing_graph is None or existing_graph.number_of_nodes() == 0:
            graph = self.build(knowledge_candidates, chunks=chunks, max_nodes=max_nodes)
            return graph, graph.number_of_nodes()

        # 候选清洗（与 build 相同口径：去重、限量、保序）
        candidates, seen = [], set()
        for k in (knowledge_candidates or [])[:max_nodes]:
            k = str(k).strip()
            if k and k not in seen:
                seen.add(k)
                candidates.append(k)
        if not candidates:
            self.logger.warning("没有知识点候选，跳过图谱构建")
            return existing_graph.copy(), 0

        # 找出真正的新知识点（旧节点直接复用，不重复调 AI）
        existing_names = {str(n) for n in existing_graph.nodes}
        new_names = [c for c in candidates if c not in existing_names]
        if not new_names:
            self.logger.info("候选知识点均已存在于图谱，无需增量更新")
            return existing_graph.copy(), 0

        # 第 1 步：只为新知识点补全属性（旧节点的描述/难度从已有图谱原样继承）
        new_nodes = self._enrich_nodes(new_names, chunks)
        if not new_nodes:
            self.logger.warning("新知识点属性补全失败，返回原图谱")
            return existing_graph.copy(), 0

        # 第 2 步：只分析与新节点相关的先修关系（旧-旧边从已有图谱复制，不重复分析）
        old_nodes = [{"name": str(n),
                      "description": str((attrs or {}).get("description") or ""),
                      "difficulty": str((attrs or {}).get("difficulty") or "中等")}
                     for n, attrs in existing_graph.nodes(data=True)]
        deps = self._analyze_dependencies(old_nodes + new_nodes, chunks, only_with=new_names)

        # 第 3 步：组装 = 旧图拷贝 + 新节点 + 涉及新节点的边
        graph = existing_graph.copy()
        for node in new_nodes:
            graph.add_node(node["name"], description=node["description"],
                           difficulty=node["difficulty"])
        for dep in deps:
            graph.add_edge(dep["from"], dep["to"], type=dep["type"])
        self.logger.info("增量构建完成：新增 %d 个节点，现 %d 节点 / %d 条关系",
                         len(new_nodes), graph.number_of_nodes(), graph.number_of_edges())
        return graph, len(new_nodes)

    # ---------- 第 1 步：知识点属性补全 ----------
    def _enrich_nodes(self, candidates, chunks=None):
        """
        调用 DeepSeek 为每个知识点生成描述与难度。
        :return: [{"name": "", "description": "", "difficulty": "基础/中等/困难"}]
        """
        excerpt = "\n".join(c["text"][:300] for c in (chunks or [])[:4]) or "（无课件摘录）"
        prompt = f"""请为下面的每个知识点补充学习资料属性。

知识点列表：
{json.dumps(candidates, ensure_ascii=False)}

课件摘录（供参考）：
{excerpt}

严格要求：
1. name 必须与知识点列表中的名称完全一致，不要增删改写；
2. description 为不超过 40 字的学习向简介；
3. difficulty 只能是"基础"、"中等"、"困难"三选一（按学习掌握该知识点的难度判断）；
4. 只输出 JSON，格式如下，不要输出任何其他文字：
{{"knowledge_points": [{{"name": "知识点", "description": "简介", "difficulty": "中等"}}]}}"""
        data = self.chat_json(prompt, temperature=0.3)
        return self._validate_nodes(data, candidates)

    # ---------- 第 2 步：先修关系分析 ----------
    def _analyze_dependencies(self, nodes, chunks=None, only_with=None):
        """
        调用 DeepSeek 分析知识点之间的先修关系。
        :param only_with: 新知识点名称列表（增量模式）。传入后只要求 AI 输出
                          与新知识点相关的依赖（旧-旧关系由已有图谱保留，不重复分析）
        :return: [{"from": "知识点A", "to": "知识点B", "type": "先修"}]
        """
        names = [n["name"] for n in nodes]
        desc = "；".join(f"{n['name']}（难度：{n['difficulty']}）" for n in nodes)
        excerpt = "\n".join(c["text"][:300] for c in (chunks or [])[:4]) or "（无课件摘录）"
        # 增量模式的额外约束：至少一端是新知识点，避免 AI 把旧关系重新输出一遍（省 token）
        focus = ""
        if only_with:
            focus = f"""
本次重点（增量分析）：{json.dumps(only_with, ensure_ascii=False)}
5. 只输出与上述新知识点相关的依赖关系（新-新、旧->新、新->旧均可）；
   旧知识点之间的已有关系已经确定，不要重复输出；"""
        prompt = f"""请分析下面知识点之间的先修关系（学习顺序依赖：先学 A 才能学 B，记为 A -> B）。

知识点及难度：
{desc}

课件摘录（供参考）：
{excerpt}

严格要求：
1. from 和 to 必须是上面列表中出现过的知识点名称，且 from != to；
2. type 只能是"先修"；
3. 只标注确实存在学习依赖的关系，不要为了凑数硬造；
4. 只输出 JSON，格式如下，不要输出任何其他文字：
{{"dependencies": [{{"from": "知识点A", "to": "知识点B", "type": "先修"}}]}}{focus}"""
        data = self.chat_json(prompt, temperature=0.3)
        return self._validate_edges(data, names)

    # ---------- 结果校验清洗（静态方法，便于单测） ----------
    @staticmethod
    def _validate_nodes(data, candidates):
        """
        校验清洗节点数据：
          - name 必须在候选列表中（防止 AI 改写名称导致与依赖分析对不上）
          - 难度限定三档，非法值回退为"中等"
          - 按候选顺序去重
        """
        if not isinstance(data, dict):
            return []
        raw = data.get("knowledge_points")
        valid = set(candidates)
        nodes, seen = [], set()
        for n in raw if isinstance(raw, list) else []:
            try:
                name = str(n["name"]).strip()
                if name not in valid or name in seen:
                    continue
                seen.add(name)
                diff = str(n.get("difficulty", "中等")).strip()
                nodes.append({
                    "name": name,
                    "description": str(n.get("description", "")).strip()[:60],
                    "difficulty": diff if diff in VALID_DIFFICULTIES else "中等",
                })
            except (KeyError, TypeError):
                continue
        return nodes

    @staticmethod
    def _validate_edges(data, node_names):
        """
        校验清洗边数据：
          - from/to 必须都是已知节点（与第 1 步产出的节点集对齐）
          - 过滤自环边；type 限定"先修"
        """
        if not isinstance(data, dict):
            return []
        raw = data.get("dependencies")
        valid = set(node_names)
        edges = []
        for e in raw if isinstance(raw, list) else []:
            try:
                src = str(e["from"]).strip()
                dst = str(e["to"]).strip()
                typ = str(e.get("type", "先修")).strip()
                if src in valid and dst in valid and src != dst and typ in VALID_TYPES:
                    edges.append({"from": src, "to": dst, "type": typ})
            except (KeyError, TypeError):
                continue
        return edges
