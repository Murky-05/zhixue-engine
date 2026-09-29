# -*- coding: utf-8 -*-
"""
EvalAgent —— 学习评估智能体（前测-后测-学习增益版）
====================================================
职责：用教育测量中经典的"前测 -> 复习 -> 后测"范式量化学习效果，
     计算归一化学习增益 ALG 并生成评估报告，结果存入 SQLite。

使用流程：
    1. pre_test()            学习前出题（前测卷）
    2. grade(quiz, answers)  学生作答后本地判分 -> pre_score
    3. （学生复习课件 / 学习路径）
    4. post_test(pre_quiz)   学习后出平行卷（同考点同难度、不同题面）
    5. grade()               判分 -> post_score
    6. calc_alg()            ALG = (后测 - 前测) / (满分 - 前测)
    7. generate_report()     AI 解读增益、生成报告并存入 SQLite

ALG（Hake 归一化学习增益）解读标准（教育测量通用约定）：
    ALG >= 0.7            高增益
    0.3 <= ALG < 0.7      中增益
    ALG < 0.3             低增益（后测 < 前测 即为退步）
    前测已满分（无提升空间）时定义为 1.0

返回格式（generate_report 的返回值）：
    {"alg": 0.5, "pre_score": 40, "post_score": 70, "full_score": 100, "report": "..."}
"""

import json
import logging

from .base_agent import BaseAgent
from .diagnosis_agent import DiagnosisAgent   # 复用其单题 JSON 校验逻辑
from utils.db import save_eval                # 评估结果统一入库（utils/db.py）


class EvalAgent(BaseAgent):
    """学习评估智能体：前测/后测出题、判分、ALG 增益计算与报告存库"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ================= 出题 =================
    def pre_test(self, graph=None, knowledge_candidates=None, n=3):
        """
        生成前测题目（学习前摸底）。
        :param graph: 知识图谱（优先按节点度数选核心知识点）
        :param knowledge_candidates: 知识点候选列表（无图谱时兜底）
        :param n: 题目数量
        :return: 题目列表 [{"knowledge_point", "question", "options",
                 "answer", "explain"}]；无可用知识点返回 []
        """
        points = self._pick_points(graph, knowledge_candidates, n)
        return self._make_quiz(points, pre_quiz=None)

    def post_test(self, pre_quiz=None, graph=None, knowledge_candidates=None, n=3):
        """
        生成后测题目（与前测平行的卷子：同考点同难度、不同题面）。
        :param pre_quiz: 前测题目列表——后测考点优先沿用前测，保证前后可比
        """
        if pre_quiz:
            points = [q["knowledge_point"] for q in pre_quiz]   # 与前测同考点
        else:
            points = self._pick_points(graph, knowledge_candidates, n)
        return self._make_quiz(points, pre_quiz=pre_quiz)

    @staticmethod
    def _pick_points(graph, knowledge_candidates, n=3):
        """选考点：优先知识图谱核心点（按度数排序），否则用知识点候选列表"""
        if graph is not None and graph.number_of_nodes() > 0:
            ranked = sorted(graph.degree, key=lambda x: (-x[1], str(x[0])))
            return [str(name) for name, _ in ranked[:n]]
        return [str(k) for k in (knowledge_candidates or [])][:n]

    def _make_quiz(self, points, pre_quiz=None):
        """为核心知识点逐个出题；pre_quiz 非空时生成平行后测卷"""
        if not points:
            self.logger.warning("没有可用知识点，无法出题")
            return []
        pre_map = {q["knowledge_point"]: q for q in (pre_quiz or [])}
        questions = []
        for kp in points:
            data = self.chat_json(self._quiz_prompt(kp, pre_map.get(kp)), temperature=0.5)
            q = DiagnosisAgent._validate_question(data, kp)   # 复用诊断 Agent 的题目校验
            if q:
                questions.append(q)
        return questions

    @staticmethod
    def _quiz_prompt(kp, pre_q=None):
        """构造单题提示词；pre_q 为该知识点的前测题（后测平行卷参照）"""
        if pre_q:
            old = json.dumps(pre_q, ensure_ascii=False)
            return f"""请为知识点「{kp}」出 1 道后测题（平行卷）。

前测原题：{old}

严格要求：
1. 考点与难度与前测题相当，但题干情境和选项必须与原题不同（避免记忆效应）；
2. options 为 4 个选项（键 A/B/C/D），answer 为正确选项字母，explain 为一句话解析；
3. 只输出 JSON：
{{"question": "题干", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "B", "explain": "解析"}}"""
        return f"""请为知识点「{kp}」出 1 道单项选择题（前测摸底）。

严格要求：
1. 题干清晰，4 个选项，干扰项有迷惑性但唯一正确；
2. options 为 4 个选项（键 A/B/C/D），answer 为正确选项字母，explain 为一句话解析；
3. 只输出 JSON：
{{"question": "题干", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "B", "explain": "解析"}}"""

    # ================= 判分与增益计算 =================
    @staticmethod
    def grade(quiz, answers):
        """本地判分：返回答对题数（answers 与 quiz 按位置对齐）"""
        correct = 0
        for q, a in zip(quiz, answers):
            yours = str(a or "").strip().upper()[:1]
            if yours and yours == str(q.get("answer", "")).strip().upper()[:1]:
                correct += 1
        return correct

    @staticmethod
    def calc_alg(pre_score, post_score, full_score):
        """
        计算归一化学习增益 ALG = (后测 - 前测) / (满分 - 前测)。
        - 分母 <= 0（前测已满分）时定义为 1.0：无提升空间即完全掌握；
        - 分数先裁剪到 [0, 满分]，防御异常输入；
        - 后测 < 前测时 ALG 为负（真实反映退步，不做截断）。
        """
        full = float(full_score)
        pre = max(0.0, min(full, float(pre_score)))
        post = max(0.0, min(full, float(post_score)))
        denom = full - pre
        if denom <= 0:
            return 1.0
        return (post - pre) / denom

    @staticmethod
    def _gain_level(alg):
        """Hake 增益等级：>=0.7 高增益；0.3~0.7 中增益；<0.3 低增益"""
        if alg >= 0.7:
            return "高增益"
        if alg >= 0.3:
            return "中增益"
        return "低增益"

    # ================= 评估报告 =================
    def generate_report(self, pre_score, post_score, full_score,
                        file_id=None, pre_quiz=None, post_quiz=None, user_name="访客"):
        """
        生成评估报告并存入 SQLite（经 utils/db.py 的 save_eval）。
        :param pre_score: 前测得分
        :param post_score: 后测得分
        :param full_score: 满分（app.py 传题数，每题 1 分）
        :param file_id: 当前课件在 files 表中的记录 id（缺失时结果不入库）
        :param pre_quiz / post_quiz: 前后测题目（提供考察知识点依据，可选）
        :return: {"alg", "pre_score", "post_score", "full_score", "report"}
        """
        # 1) 分数裁剪 + 增益计算
        full = int(full_score)
        pre = max(0, min(full, int(pre_score)))
        post = max(0, min(full, int(post_score)))
        alg = self.calc_alg(pre, post, full)
        level = self._gain_level(alg)

        # 2) AI 生成报告（失败降级为本地模板报告）
        kps = [q["knowledge_point"] for q in (pre_quiz or [])]
        report = (self._ai_report(pre, post, full, alg, level, kps)
                  or self._fallback_report(pre, post, full, alg, level))

        # 3) 组装结果并存库（eval 表只存三项数值；报告文本由页面会话展示）
        data = {
            "alg": round(alg, 4),
            "pre_score": pre,
            "post_score": post,
            "full_score": full,
            "report": report,
        }
        self._save_to_db(file_id, pre, post, data["alg"], data["report"], user_name=user_name)
        return data

    def _ai_report(self, pre, post, full, alg, level, kps):
        """调用 DeepSeek 生成 Markdown 评估报告；失败返回 None"""
        points_text = "、".join(kps) if kps else "（未提供知识点清单）"
        prompt = f"""你是学习评估专家。学生完成了一次"前测-复习-后测"循环：
- 满分 {full} 分，前测 {pre} 分，后测 {post} 分；
- 归一化学习增益 ALG = {alg:.2f}（{level}；ALG = (后测-前测)/(满分-前测)）；
- 考察知识点：{points_text}

请输出 Markdown 评估报告：
1. 用 2-3 句解读学习增益（是否有效进步、幅度评价）；
2. 给出 2-3 条下一步学习建议；
3. 语言鼓励但客观，总长度 120 字以内。"""
        return self.chat(prompt, system="你是学习评估专家。", temperature=0.4)

    @staticmethod
    def _fallback_report(pre, post, full, alg, level):
        """AI 失败时的本地模板报告，保证评估流程永不中断"""
        if post < pre:
            trend = "后测低于前测，可能存在临场失误或知识遗忘，建议重新复习后再测一次"
        elif alg >= 0.7:
            trend = "学习效果显著，请保持当前学习方法"
        elif alg >= 0.3:
            trend = "取得了中等幅度的进步，请针对薄弱点继续巩固"
        else:
            trend = "进步幅度有限，建议调整复习策略并重点攻克薄弱知识点"
        return (f"前测 {pre} 分，后测 {post} 分（满分 {full}），"
                f"学习增益 ALG={alg:.2f}，属于{level}。{trend}。")

    # ================= SQLite 存库 =================
    def _save_to_db(self, file_id, pre_score, post_score, alg, report=None, user_name="访客"):
        """
        把评估结果写入 SQLite eval 表（统一走 utils/db.py 的 save_eval）。
        缺少 file_id（如旧会话课件未入库）时跳过写库，只记录日志，不影响报告返回。
        """
        if file_id is None:
            self.logger.warning("缺少 file_id，评估结果未写入数据库")
            return
        try:
            save_eval(file_id, pre_score, post_score, alg, report, user_name=user_name)   # 报告文本一并入库，供历史回溯
            self.logger.info("评估结果已写入学习记录数据库（file_id=%s，user=%s）", file_id, user_name)
        except Exception as e:   # 数据库异常不应中断评估报告的返回
            self.logger.warning("评估结果写入数据库失败：%s", e)
