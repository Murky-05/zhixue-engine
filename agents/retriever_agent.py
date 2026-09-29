# -*- coding: utf-8 -*-
"""
RetrieverAgent —— 混合检索智能体（向量检索 + 图谱检索 + 重排序）
================================================================
职责：把用户的自然语言问题映射到最相关的课件片段，同时给出图谱维度的
     相关知识点，为 TutorAgent 提供更高质量的上下文。

检索流程（三步）：
    1. 向量召回：问题向量化（由 IndexAgent 完成），与全部段落计算
       余弦相似度（numpy 点积，向量均已 L2 归一化），取 Top-5；
    2. 图谱扩展：从问题中匹配知识图谱节点（关键词命中），对命中节点
       提取 2 跳子图，得到「命中 + 关联」知识点集合；
    3. 重排序：最终得分 = 向量得分 × 0.7 + 图谱匹配得分 × 0.3，取 Top-3。

图谱匹配得分：片段文本中包含的相关知识点个数（归一化到 0~1，
最好片段记 1.0）。图谱为空或问题无知识点命中时，图谱得分全为 0，
自动退化为纯向量检索，保证流程永不中断。

返回格式：
    {
        "chunks": [{"id", "text", "score", "pages"}, ...],  # 重排序后的 Top-3 片段
        #   pages = 片段所在 PDF 页码列表（非 PDF 为 None），供 AI 回答标注来源页码
        "knowledge_points": [                        # 图谱命中的知识点（命中在前）
            {"name", "description", "difficulty", "match_type"},
            ...
        ]
    }
"""

import networkx as nx

# ---------- 重排序权重与召回配置 ----------
VECTOR_WEIGHT = 0.7   # 向量相似度权重
GRAPH_WEIGHT = 0.3    # 图谱匹配权重（两者之和为 1）
VECTOR_TOP_K = 5      # 第一轮向量召回数量（重排序前）
MAX_HOP = 2           # 图谱扩展跳数：命中节点的 2 跳子图


class RetrieverAgent:
    """混合检索智能体：向量检索保底，知识图谱检索增强"""

    def __init__(self, index):
        """
        :param index: 已 build 的 IndexAgent 实例（内含向量矩阵与段落原文）
        """
        self.index = index

    # ---------- 主入口 ----------
    def retrieve(self, query, top_k=3, graph=None, min_score=0.0):
        """
        混合检索：向量召回 + 图谱扩展 + 重排序。
        :param query: 用户问题
        :param top_k: 最终返回的片段数量（默认 3）
        :param graph: OntologyAgent 构建的知识图谱（networkx DiGraph），
                      可为 None（图谱尚未生成时退化为纯向量检索）
        :param min_score: 置信度阈值（0~1）：重排序得分低于该值的片段不返回，
                          作为"回答依据的相关度底线"（开发者端 Agent 配置页可调）；
                          全部片段低于阈值时返回空列表，由调用方决定兜底文案
        :return: {"chunks": [...], "knowledge_points": [...]}
        """
        # ---- 第 1 步：向量召回 Top-5 ----
        # IndexAgent 内部用 numpy 把问题向量化并与全段落做余弦相似度（点积），
        # 返回 [(段落dict, 相似度)]，这里转成统一的候选结构
        vec_hits = self.index.search(query, top_k=VECTOR_TOP_K)
        candidates = [
            {"id": c["id"], "text": c["text"], "vscore": float(score), "pages": c.get("pages")}
            for c, score in vec_hits
        ]
        # 零分补齐：TF-IDF 只返回相似度>0 的片段，命中不足 top_k 时用
        # 未命中的段落补位（向量分 0），它们仍可凭图谱分在重排序中胜出，
        # 保证 TutorAgent 拿满 top_k 个上下文
        if len(candidates) < top_k and getattr(self.index, "chunks", None):
            chosen = {c["id"] for c in candidates}
            for c in self.index.chunks:
                if len(candidates) >= top_k:
                    break
                if c["id"] not in chosen:
                    candidates.append({"id": c["id"], "text": c["text"],
                                       "vscore": 0.0, "pages": c.get("pages")})
                    chosen.add(c["id"])

        # ---- 第 2 步：图谱检索（问题关键词 -> 命中节点 -> 2 跳子图） ----
        knowledge_points, related_names = [], []
        if graph is not None and graph.number_of_nodes() > 0:
            seeds = self._match_keywords(query, graph)      # 问题中命中的知识点
            if seeds:
                knowledge_points, related_names = self._expand_subgraph(seeds, graph)

        # ---- 第 3 步：重排序（0.7 × 向量分 + 0.3 × 图谱分） ----
        # 图谱匹配得分：每个候选片段包含的相关知识点个数，除以候选中最大值归一化
        hit_counts = [
            sum(1 for name in related_names if name in c["text"])
            for c in candidates
        ]
        max_hits = max(hit_counts) if hit_counts else 0
        for cand, hits in zip(candidates, hit_counts):
            gscore = (hits / max_hits) if max_hits else 0.0   # 无图谱命中时全为 0
            cand["score"] = VECTOR_WEIGHT * cand["vscore"] + GRAPH_WEIGHT * gscore

        # 按最终得分降序取 Top-3，再按置信度阈值过滤：
        # 低于 min_score 的片段不作为回答依据（输出字段与 TutorAgent 的输入约定保持一致）
        candidates.sort(key=lambda x: -x["score"])
        chunks = [
            {"id": c["id"], "text": c["text"], "score": round(c["score"], 4),
             "pages": c.get("pages")}   # PDF 页码列表（非 PDF 为 None），供来源标注
            for c in candidates[:top_k] if c["score"] >= min_score
        ]
        return {"chunks": chunks, "knowledge_points": knowledge_points}

    # ---------- 图谱检索：问题关键词匹配 ----------
    @staticmethod
    def _match_keywords(query, graph):
        """
        从问题中匹配知识图谱节点（无需分词库的中文关键词匹配）：
          - 正向：节点名出现在问题里（如问题含「光合作用」）；
          - 反向：问题片段出现在节点名里（如问题「光合」命中节点「光合作用」）。
        单字节点名容易误命中，直接跳过。
        :return: 命中的节点名列表（可能为空）
        """
        q = str(query).strip()
        seeds = []
        for name in graph.nodes:
            n = str(name).strip()
            if len(n) < 2:
                continue
            if n in q or (len(q) >= 2 and q in n):
                seeds.append(name)
        return seeds

    # ---------- 图谱检索：提取 2 跳子图 ----------
    @staticmethod
    def _expand_subgraph(seeds, graph):
        """
        以命中节点为中心提取 2 跳子图，收集子图内全部知识点。
        用无向视图扩展：先修前置（上游）与后续知识（下游）都算相关。
        :return: (knowledge_points, related_names)
                 knowledge_points 按离命中节点的跳数升序（命中在最前），
                 每项 {"name", "description", "difficulty", "match_type"}
        """
        und = graph.to_undirected()   # 无向视图：双向都能扩展
        dist = {}                     # 节点 -> 到最近命中节点的跳数
        for seed in seeds:
            lengths = nx.single_source_shortest_path_length(und, seed, cutoff=MAX_HOP)
            for node, d in lengths.items():
                if node not in dist or d < dist[node]:
                    dist[node] = d

        knowledge_points, related_names = [], []
        for node, d in sorted(dist.items(), key=lambda x: x[1]):   # 跳数升序
            attrs = graph.nodes[node]
            knowledge_points.append({
                "name": str(node),
                "description": attrs.get("description", ""),
                "difficulty": attrs.get("difficulty", "中等"),
                "match_type": "命中" if d == 0 else f"{d}跳关联",
            })
            related_names.append(str(node))
        return knowledge_points, related_names
