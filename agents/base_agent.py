# -*- coding: utf-8 -*-
"""
BaseAgent —— 所有智能体的公共基类
=================================
统一职责：
1. 从项目根目录 .env 读取 DEEPSEEK_API_KEY，创建 OpenAI 兼容的 DeepSeek 客户端
2. chat()：普通对话调用，返回文本（失败返回 None，由调用方决定降级策略）
3. chat_json()：要求模型返回 JSON 并稳健解析（兼容 ```json 代码块、前后缀文字）

子类只需关注各自的提示词与业务逻辑，不必重复处理密钥与请求代码。
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

# 用量入库：记录每次调用的 token 消耗与估算费用（"用量信息"页数据源）
from utils.auth import get_current_user   # 当前会话用户 {"user_name", "role"}
from utils.db import log_event, save_api_usage
# 文件日志：AI 调用成功流水 -> business.log（INFO）；报错堆栈 -> error.log（ERROR）
from utils.log_config import business_logger, error_logger

# .env 位于 agents/ 的上一级（项目根目录）；在导入本模块时即加载
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 统一日志格式：后端模块使用 logging，不依赖任何 UI 框架
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ========== 计费单价（元 / 百万 tokens）==========
# DeepSeek 定价口径：输入 1 元/百万 tokens，输出 2 元/百万 tokens
PRICE_PROMPT_PER_M = 1.0
PRICE_COMPLETION_PER_M = 2.0

# ========== 全局异常处理约定：用户端 / 开发者端分离 ==========
# 用户端文案：只描述"发生了什么、该做什么"，绝不出现 余额/充值/DeepSeek 等字眼
USER_MSG_SERVICE = "AI服务暂时不可用，请稍后重试"
USER_MSG_NETWORK = "网络连接不太稳定，请检查网络后稍后重试"
USER_MSG_BUSY = "AI 服务繁忙，请稍后再试"
# 开发者端：详细错误与堆栈一律写入 logs/error.log 并打印终端标签，绝不上前端页面

# ========== 统一错误码（API 网关封装：前端只认错误码，不见原始 Exception） ==========
ERR_402 = "ERR_402"                # 余额不足
ERR_TIMEOUT = "ERR_TIMEOUT"        # 请求超时
ERR_RATE_LIMIT = "ERR_RATE_LIMIT"  # 触发限流（HTTP 429）
ERR_NETWORK = "ERR_NETWORK"        # 网络连接失败（扩展码：与超时同属网络类暂时性错误）
ERR_UNKNOWN = "ERR_UNKNOWN"        # 未知错误（其他 HTTP 状态码 / 返回内容无法解析等）

# 错误码 -> 用户可读文案（前端文案的唯一来源；调用方按码取文案，绝不抛原始异常）
ERROR_MESSAGES = {
    ERR_402: USER_MSG_SERVICE,
    ERR_TIMEOUT: USER_MSG_NETWORK,
    ERR_RATE_LIMIT: USER_MSG_BUSY,
    ERR_NETWORK: USER_MSG_NETWORK,
    ERR_UNKNOWN: USER_MSG_SERVICE,
}


class APIGateway:
    """
    DeepSeek API 网关（所有底层调用的唯一拦截入口，"API网关封装"模式）：
      1. 指数退避重试：网络超时 / 连接失败 / 429 限流自动重试 3 次（间隔 1s/2s/4s）；
      2. 统一错误码：把 openai SDK 的各类异常归类为 ERR_* 错误码，
         调用方（BaseAgent / 前端）只见错误码与映射文案，永不接触原始 Exception；
      3. 调用审计：每一次请求的耗时与状态（SUCCESS / ERR_*）写入 logs/business.log，
         最终失败的完整堆栈写入 logs/error.log。
    """

    RETRY_DELAYS = (None, 1, 2, 4)   # 首请求不等待；其后按 1s/2s/4s 指数退避

    def __init__(self, client):
        self.client = client

    @staticmethod
    def _classify_status(e):
        """把 APIStatusError（非 2xx 响应）归类为统一错误码；返回 (错误码, HTTP状态码)"""
        status = getattr(e, "status_code", 0) or 0
        if status == 402:
            return ERR_402, status
        if status == 429:
            return ERR_RATE_LIMIT, status
        return ERR_UNKNOWN, status

    def invoke(self, messages, model, temperature):
        """
        执行一次经网关的 API 调用（重试 / 计时 / 状态审计全部在此收口）。
        :param messages: OpenAI 格式的消息列表
        :return: (code, resp) —— 成功: (None, 原始响应对象)；失败: (ERR_*, None)
        """
        last_exc, last_code = None, ERR_UNKNOWN
        for i, delay in enumerate(self.RETRY_DELAYS):
            if delay is not None:
                print(f"[RETRY {i}/3] {last_code}，{delay}s 后重试...")
                time.sleep(delay)
            start = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature)
                duration = time.perf_counter() - start
                business_logger.info("[API网关] status=SUCCESS model=%s 耗时=%.2fs 重试次数=%d",
                                     model, duration, i)
                if i > 0:
                    print(f"[RETRY OK] 第 {i} 次重试成功")
                return None, resp
            except openai.APIStatusError as e:
                duration = time.perf_counter() - start
                code, status = self._classify_status(e)
                business_logger.info("[API网关] status=%s model=%s 耗时=%.2fs 重试次数=%d (HTTP %s)",
                                     code, model, duration, i, status)
                if code == ERR_RATE_LIMIT:   # 限流：暂时性错误，退避后重试
                    last_exc, last_code = e, code
                    continue
                return self._final(code, e)   # 402 等不可重试错误：立即收尾
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                duration = time.perf_counter() - start
                code = ERR_TIMEOUT if isinstance(e, openai.APITimeoutError) else ERR_NETWORK
                business_logger.info("[API网关] status=%s model=%s 耗时=%.2fs 重试次数=%d",
                                     code, model, duration, i)
                last_exc, last_code = e, code   # 网络类暂时性错误：退避后重试
                continue
            except Exception as e:
                duration = time.perf_counter() - start
                business_logger.info("[API网关] status=%s model=%s 耗时=%.2fs 重试次数=%d",
                                     ERR_UNKNOWN, model, duration, i)
                return self._final(ERR_UNKNOWN, e)
        # 重试额度用尽：按最后一次错误码统一收尾
        return self._final(last_code, last_exc)

    def _final(self, code, err):
        """最终失败收尾：终端一行标签（开发者实时观察）+ error.log 完整堆栈（事后追溯）"""
        print(f"[ERROR {code}] {type(err).__name__ if err is not None else 'Unknown'}")
        error_logger.error("[API网关] 最终失败 status=%s", code, exc_info=err is not None)
        return code, None


class BaseAgent:
    """智能体基类：封装 DeepSeek 客户端、对话调用与 JSON 解析"""

    def __init__(self, api_key=None, model="deepseek-chat", temperature=0.7):
        """
        :param api_key: DeepSeek API Key；不传则自动从 .env 读取 DEEPSEEK_API_KEY
        :param model: 模型名，默认 deepseek-chat
        :param temperature: 默认采样温度（越低越严谨，越高越发散）
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("未检测到 DEEPSEEK_API_KEY，请在项目根目录 .env 中配置")
        # 使用 OpenAI SDK 兼容方式调用 DeepSeek
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        # API 网关：所有底层调用的统一拦截入口（重试/计时/错误码/审计）
        self.gateway = APIGateway(self.client)
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)
        # 最近一次调用失败的分类信息 {"code": ERR_*, "message": 映射文案}；成功时复位为 None。
        # code 是统一错误码，message 是给用户端显示的安全文案（app.py 通过 ai_fail_hint 展示）
        self.last_error = None

    def chat(self, prompt, system=None, temperature=None):
        """
        调用 DeepSeek 对话接口（"API网关封装"模式的统一入口，所有 Agent 的调用都经过这里）。

        —— 网关统一拦截 ——
          1. 指数退避重试：网络超时 / 连接失败 / 429 限流自动重试 3 次（间隔 1s/2s/4s），
             不可重试错误（402 余额不足等）立即收尾，不做无谓等待；
          2. 统一错误码：网关把所有异常归类为 ERR_* 错误码；本方法把错误码映射为
             用户可读文案（ERROR_MESSAGES）写入 self.last_error——前端经 ai_fail_hint
             只展示映射文案，绝不抛出/展示原始 Exception；
          3. 调用审计：每次请求的耗时与状态（SUCCESS / ERR_*）由网关写入
             logs/business.log，最终失败的完整堆栈写入 logs/error.log。
        :return: 模型回复文本；失败返回 None（不抛异常，页面永不崩溃）
        """
        messages = [
            {"role": "system", "content": system or "你是一个专业的智能学习助手。"},
            {"role": "user", "content": prompt},
        ]
        # 全部底层调用经网关拦截：重试、计时、状态审计、错误码分类都在 invoke 内完成
        code, resp = self.gateway.invoke(
            messages, self.model,
            self.temperature if temperature is None else temperature)
        if code is not None:
            # 失败：错误码 -> 用户可读文案（前端绝不接触原始 Exception），并落库 event_log
            self.last_error = {"code": code,
                               "message": ERROR_MESSAGES.get(code, ERROR_MESSAGES[ERR_UNKNOWN])}
            self._log_api_error(code, f"gateway status={code}")
            return None
        # 成功：复位错误状态，记录 token 用量，返回回复文本
        self.last_error = None
        self._record_usage(resp)
        business_logger.info("[AI调用] %s 成功 (model=%s)", self.__class__.__name__, self.model)
        return resp.choices[0].message.content

    def _log_api_error(self, code, err):
        """
        把 API 错误持久化到 event_log 表（event="api_error"），
        供开发者端"系统日志"页查看最近错误。
        agent.last_error 只是当前会话的内存态，跨会话不可见，
        因此错误发生时同步落库。日志失败不影响主流程。
        """
        try:
            user = get_current_user()
            log_event("api_error", f"[{code}] {type(err).__name__}: {err}",
                      user_name=user["user_name"], role=user["role"])
        except Exception:
            pass   # 日志写入失败绝不影响回答流程

    def _record_usage(self, resp):
        """
        读取 response.usage 并把本次调用的 token 消耗与估算费用写入 api_usage 表。
        计费口径：输入 1 元/百万 tokens，输出 2 元/百万 tokens。
        用户/角色归属：从当前 Streamlit 会话读取（bare 模式兜底"访客"/student）。
        用量记录失败只写日志、绝不影响回答返回（主流程优先）。
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            return   # 响应未携带 usage（异常情况），跳过记录
        try:
            prompt_t = usage.prompt_tokens or 0
            completion_t = usage.completion_tokens or 0
            total_t = usage.total_tokens or (prompt_t + completion_t)
            # 估算费用（元）：输入单价×输入量 + 输出单价×输出量
            cost = (prompt_t / 1_000_000 * PRICE_PROMPT_PER_M
                    + completion_t / 1_000_000 * PRICE_COMPLETION_PER_M)
            user = get_current_user()   # {"user_name", "role"}
            save_api_usage(prompt_t, completion_t, total_t, round(cost, 6),
                           user_name=user["user_name"], role=user["role"])
            self.logger.info("本次调用用量：输入 %d + 输出 %d = %d tokens，估算费用 %.6f 元",
                             prompt_t, completion_t, total_t, cost)
        except Exception as e:
            self.logger.warning("API 用量记录入库失败（不影响本次回答）：%s", e)

    def chat_json(self, prompt, system=None, temperature=None):
        """调用并解析 JSON；返回 Python 对象（dict/list），失败返回 None"""
        raw = self.chat(prompt, system=system or "你是一个只输出严格 JSON 的助手。", temperature=temperature)
        if raw is None:
            return None   # chat 已失败：last_error 已由 chat() 设置
        data = self.extract_json(raw)
        if data is None:
            # 模型回复了但内容不是合法 JSON：按统一错误码 ERR_UNKNOWN 落账
            print("[ERROR json] Model returned non-JSON content")
            business_logger.info("[API网关] status=ERR_PARSE agent=%s", self.__class__.__name__)
            error_logger.error("[AI调用失败] 模型返回内容不是合法 JSON (agent=%s)", self.__class__.__name__)
            self.last_error = {"code": ERR_UNKNOWN, "message": ERROR_MESSAGES[ERR_UNKNOWN]}
        return data

    @staticmethod
    def extract_json(text):
        """
        从模型回复中稳健地提取 JSON 对象或数组。
        兼容：```json 代码块包裹、JSON 前后带解释文字等情况。
        """
        if not text:
            return None
        text = re.sub(r"```(?:json)?", "", text)  # 去掉代码块标记
        # 优先匹配 JSON 对象 {...}，其次匹配数组 [...]
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
        return None
