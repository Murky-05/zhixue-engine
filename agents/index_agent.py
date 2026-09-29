# -*- coding: utf-8 -*-
"""
IndexAgent —— 本地检索索引智能体
================================
职责：把课件文本块变成「向量」，支持"问题 -> 最相关片段"的快速匹配。

向量化策略（两级）：
    1. 优先调用 DeepSeek Embedding API 把每个段落转为语义向量；
    2. 若 DeepSeek 不支持 embedding（或调用失败），自动降级为
       本地 TF-IDF（字符 bigram）向量 —— 纯 numpy 计算，零 API 成本。

核心数据（build 后即可访问，也是对外输出的"返回值"）：
    self.vectors : np.ndarray          # 每行是一个段落的向量（已 L2 归一化）
    self.id_map  : {0: "chunk_001", 1: "chunk_002", ...}   # 行号 -> 段落ID
    self.result  : {"vectors": ..., "id_map": ...}          # 规格化输出

兼容性说明：
    build() 返回 self 以支持链式调用 —— app.py 中
    `IndexAgent().build(chunks)` 的返回值会直接传给 RetrieverAgent，
    因此保留实例返回；需要规格化字典时访问 `agent.result` 即可。

多租户隔离说明：
    索引的数据源只有当前会话的课件文本块（session_state.doc["chunks"]），
    实例仅在会话内存在、不落盘、无全局共享缓存；恢复历史课件时，chunks
    同样只来自数据库中归属于当前用户的 files 记录（user_name+role 双过滤）——
    因此每个用户的检索索引天然互不可见，实现"专属知识库"隔离。
"""

import logging
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# .env 位于 agents/ 的上一级（项目根目录）；幂等加载，保证单独导入本模块也能读到密钥
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EMBED_BATCH_SIZE = 32   # 每次 embedding 请求最多携带的文本条数，避免单请求过大


class IndexAgent:
    """
    检索索引智能体：段落 -> 向量矩阵 + ID映射 -> 余弦相似度检索。
    纯本地降级模式（TF-IDF）不需要 API Key，因此不继承 BaseAgent
    （BaseAgent 在无密钥时会直接抛异常，这里需要更宽松的降级策略）。
    """

    def __init__(self, api_key=None, model="deepseek-embedding"):
        """
        :param api_key: DeepSeek API Key；不传则自动从 .env 读取（可选）
        :param model:   embedding 模型名；DeepSeek 若不支持该模型会自动降级
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_model = model

        # 仅当拿到了 API Key 才创建客户端；没有 Key 也能用 TF-IDF 正常工作
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = (OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
                       if self.api_key else None)

        # ---- build() 之后填充的核心数据 ----
        self.chunks = []      # 原始文本块 [{"id", "text"}]，检索时按行号取回
        self.vectors = None   # np.ndarray (n_chunks, dim)，行已 L2 归一化
        self.id_map = {}      # {行号: 段落ID}，如 {0: "chunk_001", ...}
        self.mode = None      # "embedding" 或 "tfidf"，记录当前向量化方式

    # ---------- 主入口 ----------
    def build(self, paragraphs):
        """
        构建索引：
          1) 统一输入格式（支持纯字符串列表或 [{"id","text"}] 列表）
          2) 优先 DeepSeek embedding，失败自动降级 TF-IDF
          3) 生成 vectors（numpy 数组）与 id_map（段落ID映射）
        :param paragraphs: 段落列表；元素为 str 或 {"id", "text"} 字典
        :return: self（链式调用）；规格化输出见 .result 属性
        """
        # 第 1 步：统一为 [{"id", "text"}] 结构，纯字符串自动编号
        self.chunks = [
            {"id": p["id"], "text": p["text"]} if isinstance(p, dict)
            else {"id": f"chunk_{i:03d}", "text": str(p)}
            for i, p in enumerate(paragraphs, start=1)
        ]
        # 第 2 步：行号 -> 段落ID 的映射（即规格中的 id_map）
        self.id_map = {i: c["id"] for i, c in enumerate(self.chunks)}

        texts = [c["text"] for c in self.chunks]
        if not texts:
            self.vectors = np.zeros((0, 1))
            self.mode = "tfidf"
            return self

        # 第 3 步：优先尝试 DeepSeek embedding；任何失败都降级 TF-IDF
        self.vectors = self._embed_all(texts)
        if self.vectors is not None:
            self.mode = "embedding"
            self.logger.info("索引构建完成（DeepSeek embedding），矩阵形状 %s", self.vectors.shape)
        else:
            self.mode = "tfidf"
            self.vectors = self._build_tfidf(texts)
            self.logger.info("索引构建完成（TF-IDF 降级），矩阵形状 %s", self.vectors.shape)
        return self

    @property
    def result(self):
        """规格化输出：{"vectors": np.array, "id_map": {0: "chunk_001", ...}}"""
        return {"vectors": self.vectors, "id_map": self.id_map}

    # ---------- 检索 ----------
    def search(self, query, top_k=3):
        """
        检索与 query 最相关的 top_k 个段落。
        :return: [(chunk_dict, score)]，按分数降序；无命中时退回开头几个块兜底
        """
        if self.vectors is None or not self.chunks:
            return []

        q = None
        if self.mode == "embedding":
            # embedding 模式：查询也要走同一个 embedding API
            qv = self._embed_all([query])
            if qv is not None:
                q = qv[0]
        else:
            # TF-IDF 模式：查询按同样的 bigram 词表转向量
            q = self._query_vector(query)

        if q is not None:
            scores = self.vectors @ q            # 行已归一化，点积即余弦相似度
            order = np.argsort(-scores)[:top_k]
            hits = [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
            if hits:
                return hits

        # 兜底：查询向量化失败（如 embedding API 中途挂了）或全部零分时，
        # 退回开头几个块，保证问答流程不中断
        self.logger.warning("检索无命中，退回开头 %d 个段落兜底", top_k)
        return [(self.chunks[i], 0.0) for i in range(min(top_k, len(self.chunks)))]

    # ---------- DeepSeek embedding ----------
    def _embed_all(self, texts):
        """
        把一批文本全部转为向量。
        :return: np.ndarray (n_texts, dim)，行已 L2 归一化；失败返回 None
        每批 EMBED_BATCH_SIZE 条分批请求；任何一批失败即整体放弃（由调用方降级）。
        """
        if self.client is None:
            return None
        rows = []
        try:
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[start:start + EMBED_BATCH_SIZE]
                resp = self.client.embeddings.create(model=self.embedding_model, input=batch)
                # 按 SDK 返回的 index 排序还原顺序，保证向量与段落一一对应
                data = sorted(resp.data, key=lambda d: d.index)
                rows.extend(d.embedding for d in data)
        except Exception as e:
            self.logger.warning("DeepSeek embedding 调用失败，将降级为 TF-IDF：%s", e)
            return None
        vec = np.array(rows, dtype=np.float32)
        return self._normalize(vec)

    # ---------- TF-IDF 降级实现 ----------
    def _build_tfidf(self, texts):
        """字符 bigram TF-IDF 矩阵（中文友好，无需分词库），行已 L2 归一化"""
        docs = [self._bigrams(t) for t in texts]

        # 建立词表：所有文档出现过的 bigram -> 列号
        # 存为实例属性，search() 阶段把查询文本转向量时要用同一张词表
        vocab = sorted({g for d in docs for g in d})
        self._tfidf_vocab = {g: j for j, g in enumerate(vocab)}
        vocab_map = self._tfidf_vocab

        # 词频矩阵 TF
        tf = np.zeros((len(docs), len(vocab)))
        for i, d in enumerate(docs):
            for g, cnt in d.items():
                tf[i, vocab_map[g]] = cnt

        # 逆文档频率 IDF（+1 平滑，避免除零）
        df = (tf > 0).sum(axis=0)
        idf = np.log((1 + len(docs)) / (1 + df)) + 1
        return self._normalize(tf * idf)

    def _query_vector(self, query):
        """把查询文本转成与 TF-IDF 矩阵同维度的归一化向量；空向量返回 None"""
        vec = np.zeros(len(self._tfidf_vocab))
        for g, cnt in self._bigrams(query).items():
            j = self._tfidf_vocab.get(g)
            if j is not None:
                vec[j] = cnt
        norm = np.linalg.norm(vec)
        return vec / norm if norm else None

    @staticmethod
    def _normalize(mat):
        """矩阵每行 L2 归一化：之后余弦相似度可直接用点积计算"""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1, norms)

    @staticmethod
    def _bigrams(text):
        """把文本切成字符二元组计数（中文友好）；单字文本退化为单字计数"""
        clean = re.sub(r"\s", "", str(text))
        if len(clean) < 2:
            return Counter({clean: 1}) if clean else Counter()
        return Counter(clean[i:i + 2] for i in range(len(clean) - 1))
