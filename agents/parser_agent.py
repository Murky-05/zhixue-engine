# -*- coding: utf-8 -*-
"""
ParserAgent —— 多格式课件解析智能体（PDF / Word / TXT / Markdown）
=================================================================
职责：把一份课件文件加工成「文本块 + 知识点候选」的结构化数据。

支持的格式（按文件扩展名自动分发到对应解析器，输出统一的文本块列表）：
    .pdf    pdfplumber 逐页提取文本
    .docx   python-docx 提取段落 + 表格文字
    .txt    自动识别编码（UTF-8 / GBK）读取纯文本
    .md     按 TXT 读取后做轻量 Markdown 清理（去标记符号，保留正文）

统一流程：
    1. 按扩展名提取全文文本
    2. 按段落切分（连续空行作为段落边界）
    3. 合并相邻短段落 / 切分过长段落，保证每块约 300-800 字
    4. 每 5 块合并一次请求抽取知识点，按顺序拆分结果；批量失败自动降级为单块调用
    5. 汇总去重，得到知识点候选列表

返回格式：
    {
        "chunks": [{"id": "chunk_001", "text": "...", "pages": [2] 或 None}, ...],
        "knowledge_candidates": ["知识点1", "知识点2", ...]
    }
    注：pages 为该文本块覆盖的 PDF 页码列表（供 AI 回答标注"来源：第X页"）；
        Word/TXT/Markdown 无页码概念，pages 为 None。

使用示例：
    from agents.parser_agent import ParserAgent
    agent = ParserAgent()
    result = agent.parse("课件.pdf")    # 也支持 .docx / .txt / .md，以及文件对象

命令行快速测试：
    python agents/parser_agent.py 课件.pdf
"""

import re
from pathlib import Path

import pdfplumber
from docx import Document as _DocxDocument   # python-docx：Word 文档解析
from pdfminer.pdfdocument import PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError

from .base_agent import BaseAgent

# ---------- 常量配置 ----------
MIN_CHUNK_SIZE = 300   # 文本块最小长度（字）
MAX_CHUNK_SIZE = 800   # 文本块最大长度（字）

# 扫描件/加密文件的统一友好提示（用户端直接展示，不含任何技术术语）
SCAN_OR_ENCRYPT_MSG = "该文档可能是扫描件或加密文件，当前不支持OCR，请更换文件"

# 支持的文件扩展名（与 app.py 上传组件的 type 列表保持一致）
SUPPORTED_EXTS = (".pdf", ".docx", ".txt", ".md")


# ========== 自定义异常：让调用方能精确区分失败原因 ==========
class ParseError(Exception):
    """PDF 解析失败（基类）：文件损坏、无法读取等"""


class EncryptedPDFError(ParseError):
    """PDF 已加密，无法解析"""


class ScannedPDFError(ParseError):
    """扫描版 PDF（纯图片、无文字层），无法直接提取文本，需要 OCR"""


class ParserAgent(BaseAgent):
    """多格式课件解析智能体：PDF/Word/TXT/MD -> 文本块 -> 知识点候选"""

    def __init__(self, api_key=None, model="deepseek-chat",
                 min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE,
                 knowledge_batch_size=5):
        """
        :param min_chunk_size: 文本块长度下限（合并短段落的目标值）
        :param max_chunk_size: 文本块长度上限（切分长段落的阈值）
        :param knowledge_batch_size: 知识点抽取时每次 API 调用携带的文本块数
        """
        super().__init__(api_key=api_key, model=model)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.knowledge_batch_size = knowledge_batch_size

    # ---------- 主入口 ----------
    def parse(self, source, progress_cb=None):
        """
        解析课件文件，返回结构化结果（= extract_chunks + extract_knowledge 的组合）。
        :param source: 文件路径（str/Path）或文件对象（如 Streamlit UploadedFile），
                       支持 .pdf / .docx / .txt / .md（按扩展名自动选择解析器）
        :param progress_cb: 可选进度回调（透传给各阶段，供 UI 实时展示）
        """
        self.logger.info("开始解析课件 ...")
        chunks = self.extract_chunks(source, progress_cb=progress_cb)
        self.logger.info("共得到 %d 个文本块，开始抽取知识点 ...", len(chunks))
        knowledge_candidates = self.extract_knowledge(chunks, progress_cb=progress_cb)
        self.logger.info("解析完成，共 %d 个知识点候选", len(knowledge_candidates))
        return {"chunks": chunks, "knowledge_candidates": knowledge_candidates}

    # ---------- 分阶段入口（供 UI 断点续传使用，parse() 即两者的组合） ----------
    def extract_chunks(self, source, progress_cb=None):
        """
        阶段 1（纯本地计算，不调 API）：提取文本 -> 按空行切段 -> 合并/切分为 300-800 字文本块。
        PDF 逐页切段并记录每块覆盖的页码（pages 字段），供 AI 回答标注"来源：第X页"。
        :param source: 文件路径或文件对象（.pdf / .docx / .txt / .md）
        :param progress_cb: 可选回调 progress_cb(done, total)——PDF 按页、其他格式整份触发
        :return: [{"id": "chunk_001", "text": ..., "pages": [2] 或 None}, ...]
        """
        text = self._extract_text(source, progress_cb=progress_cb)
        # PDF：逐页切段并携带页码（_extract_text_pdf 已把每页文本存入 last_page_texts）；
        # 其他格式无页码概念，统一 None
        page_texts = getattr(self, "last_page_texts", None)
        if page_texts:
            para_items = [(p, i)                          # (段落文本, 所属页码)
                          for i, pt in enumerate(page_texts, start=1)
                          for p in self._split_paragraphs(pt)]
        else:
            para_items = [(p, None) for p in self._split_paragraphs(text)]
        merged = self._merge_paragraphs(para_items)
        chunks = [{"id": f"chunk_{i:03d}", "text": t, "pages": pages or None}
                  for i, (t, pages) in enumerate(merged, start=1)]
        self.logger.info("文本切块完成：共 %d 个文本块", len(chunks))
        return chunks

    def extract_knowledge(self, chunks, progress_cb=None):
        """
        阶段 2（调用 DeepSeek）：对文本块每 5 块一组合并请求，抽取知识点候选并去重。
        :param chunks: extract_chunks() 的返回值
        :param progress_cb: 可选回调 progress_cb(done_batches, total_batches, n_found)
        :return: 知识点候选名称列表
        """
        candidates = self._extract_knowledge(chunks, progress_cb=progress_cb)
        self.logger.info("知识点抽取完成：共 %d 个候选", len(candidates))
        return candidates

    # ---------- 第 1 步：文本提取（按扩展名分发到对应解析器） ----------
    @staticmethod
    def _detect_ext(source):
        """
        从文件来源推断扩展名（统一小写、带点）。
        文件对象取 .name 属性（Streamlit UploadedFile / 内置文件句柄均有），路径直接取后缀。
        """
        name = getattr(source, "name", None) or str(source)
        return Path(name).suffix.lower()

    def _extract_text(self, source, progress_cb=None):
        """
        格式分发器：按扩展名调用对应的解析器，统一返回全文文本（str）。
        任何不支持的格式抛 ParseError（消息面向用户，可直接展示）。
        :param progress_cb: 可选回调 progress_cb(done, total)，透传给具体解析器
        """
        ext = self._detect_ext(source)
        self.last_page_texts = None   # 非 PDF 格式无页码概念；PDF 解析成功后会覆盖为逐页文本列表
        if ext == ".pdf":
            return self._extract_text_pdf(source, progress_cb=progress_cb)
        if ext == ".docx":
            return self._extract_text_docx(source, progress_cb=progress_cb)
        if ext == ".txt":
            return self._extract_text_txt(source, progress_cb=progress_cb)
        if ext == ".md":
            return self._extract_text_md(source, progress_cb=progress_cb)
        raise ParseError(f"暂不支持「{ext or '未知格式'}」文件，请上传 PDF / Word / TXT / Markdown 课件。")

    def _extract_text_pdf(self, pdf_source, progress_cb=None):
        """
        PDF 解析：pdfplumber 逐页提取文本，页面之间用换行拼接。
        :param progress_cb: 每提取完一页触发一次 progress_cb(已完成页数, 总页数)
        异常处理策略：
          - 需要密码的加密 PDF          -> EncryptedPDFError（友好提示）
          - 其他不支持的加密方式        -> EncryptedPDFError（友好提示）
          - 文件损坏 / 不是有效 PDF     -> ParseError
          - 文件不存在 / 无法读取       -> ParseError
          - 所有页面都提取不到文字（扫描版）-> ScannedPDFError（友好提示）
        """
        try:
            with pdfplumber.open(pdf_source) as pdf:
                n_pages = len(pdf.pages)
                page_texts = []
                for i, page in enumerate(pdf.pages, start=1):
                    page_texts.append(page.extract_text() or "")
                    if progress_cb:
                        progress_cb(i, n_pages)   # 每页回调一次，供 UI 实时更新进度条
        except PDFPasswordIncorrect:
            # PDF 设了打开密码（用户密码），pdfminer 无法用空密码解密
            raise EncryptedPDFError(SCAN_OR_ENCRYPT_MSG)
        except PDFEncryptionError:
            # 使用了不支持的加密算法等情况
            raise EncryptedPDFError(SCAN_OR_ENCRYPT_MSG)
        except PDFSyntaxError:
            # 文件结构损坏、内容不是 PDF 等
            raise ParseError("文件损坏或不是有效的 PDF 文件。")
        except OSError:
            # FileNotFoundError 等文件读取问题：用户端只显示友好提示，路径细节不外泄
            raise ParseError("文件不存在或无法读取，请检查文件后重试。")
        except (EncryptedPDFError, ScannedPDFError):
            raise   # 透传业务语义异常，避免被下方兜底逻辑吞掉
        except Exception:
            # 兜底：包装为友好文案，避免底层错误细节（含第三方库报错）泄露到用户端；
            # 原始错误与堆栈已打印到开发者终端（logger）
            self.logger.error("PDF 解析出现未知错误", exc_info=True)
            raise ParseError("PDF 解析失败，请确认文件完好后重试。")

        text = "\n".join(page_texts).strip()
        if not text:
            # 一页文字都提不出来 => 大概率是扫描版（图片型）PDF
            raise ScannedPDFError(SCAN_OR_ENCRYPT_MSG)
        # 保留每页文本（extract_chunks 逐页切段时记录页码，供 AI 回答标注来源页码）
        self.last_page_texts = page_texts
        return text

    def _extract_text_docx(self, docx_source, progress_cb=None):
        """
        Word（.docx）解析：python-docx 提取全部段落文字 + 表格单元格文字。
        注：旧版 .doc（二进制格式）不属于 docx，会走 ParseError 友好提示。
        """
        try:
            doc = _DocxDocument(docx_source)
            texts = [p.text for p in doc.paragraphs]
            # 课件常用表格排版：把表格单元格文字也纳入正文
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        texts.append(cell.text)
        except Exception:
            # 打开失败（文件损坏 / 实际是加密 docx / 伪装成 docx 的其他文件）
            self.logger.error("Word 文档解析失败", exc_info=True)
            raise ParseError("Word 文档解析失败，可能是文件加密或已损坏，请确认文件后重试。")

        if progress_cb:
            progress_cb(1, 1)   # 整份提取（无分页概念），回调保持接口统一
        text = "\n".join(t for t in texts if t and t.strip()).strip()
        if not text:
            raise ParseError("Word 文档中没有提取到文字内容，请检查文件后重试。")
        return text

    def _extract_text_txt(self, txt_source, progress_cb=None):
        """纯文本（.txt）解析：自动识别编码（UTF-8 优先，其次 GBK），失败降级为替换模式"""
        text = self._read_text_file(txt_source)
        if progress_cb:
            progress_cb(1, 1)   # 整份读取，回调保持接口统一
        if not text.strip():
            raise ParseError("该文本文件内容为空，无法解析，请检查文件。")
        return text

    def _extract_text_md(self, md_source, progress_cb=None):
        """
        Markdown（.md）解析：按 TXT 读取后做轻量清理——
        去掉代码围栏标记、标题井号、图片与链接语法、分隔线，只保留正文文字。
        代码块内容保留（可能含知识点），仅去掉围栏符号本身。
        """
        text = self._read_text_file(md_source)
        if progress_cb:
            progress_cb(1, 1)   # 整份读取，回调保持接口统一
        text = re.sub(r"```[a-zA-Z0-9_+-]*", "", text)            # 代码围栏标记
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)        # 标题井号
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # 图片 ![...](...)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # 链接 [文字](url) -> 文字
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.M)    # 分隔线 --- *** ___
        text = re.sub(r"^\s*>\s?", "", text, flags=re.M)          # 引用符号 >
        text = re.sub(r"[*_`]{1,3}", "", text)                    # 加粗/斜体/行内代码标记
        if not text.strip():
            raise ParseError("该 Markdown 文件内容为空，无法解析，请检查文件。")
        return text

    @staticmethod
    def _read_text_file(source):
        """
        读取文本类文件的原始字节并解码：
          1. 先按 UTF-8（最常见）；
          2. 失败再按 GBK（Windows 中文记事本默认编码）；
          3. 仍失败用替换模式兜底（个别乱码字符不阻断整体解析）。
        """
        try:
            raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        except OSError:
            raise ParseError("文件不存在或无法读取，请检查文件后重试。")
        finally:
            # 文件对象读完复位指针，保证流水线重跑 / 后续阶段可再次读取
            if hasattr(source, "seek"):
                try:
                    source.seek(0)
                except Exception:
                    pass
        for encoding in ("utf-8", "gbk"):
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, ValueError):
                continue
        return raw.decode("utf-8", errors="replace")   # 兜底：无法识别的编码，乱码字符替换为占位符

    # ---------- 第 2 步：段落切分 ----------
    @staticmethod
    def _split_paragraphs(text):
        """
        按段落切分：连续空行（两个及以上换行，中间可有空白字符）视为段落边界。
        同时清理 PDF 提取时常见的「硬换行」：
          - 含中文的段落：直接去掉所有空白（中文不需要空格分词）
          - 纯西文段落：换行合并为单个空格
        """
        paragraphs = []
        for raw in re.split(r"\n\s*\n+", text):
            raw = raw.strip()
            if not raw:
                continue
            if re.search(r"[\u4e00-\u9fff]", raw):   # 段落中含中文
                cleaned = re.sub(r"\s+", "", raw)
            else:                                     # 纯英文/数字等西文
                cleaned = re.sub(r"\s+", " ", raw)
            if cleaned:
                paragraphs.append(cleaned)
        return paragraphs

    # ---------- 第 3 步：合并/切分，控制块长度 ----------
    def _merge_paragraphs(self, paragraphs):
        """
        让每个文本块尽量落在 [min_chunk_size, max_chunk_size] 区间：
          1) 超过上限的段落先按句子边界切分成子段；
          2) 再贪心合并相邻短段落，凑到下限即成块；
          3) 结尾剩余的短内容若能装进最后一块（不超上限）就并入，否则独立成块。
        硬性保证：任何一块都不会超过上限；孤尾块宁短勿丢。
        :param paragraphs: [(段落文本, 页码)] 列表；页码非 PDF 时为 None
        :return: [(块文本, 页码列表)] 列表；页码列表为块内出现过的页码（升序去重），
                 无页码（非 PDF）时为空列表
        """
        # 第 1 步：把过长段落按句子切成不超过上限的子段（页码跟随原段落）
        pieces = []
        for text, page in paragraphs:
            if len(text) > self.max_chunk_size:
                pieces.extend((sub, page) for sub in self._split_long_paragraph(text))
            else:
                pieces.append((text, page))

        # 第 2 步：贪心合并相邻片段，同时聚合块内出现过的页码
        chunks, buf, buf_pages = [], "", set()
        for text, page in pieces:
            if buf and len(buf) + len(text) > self.max_chunk_size:
                chunks.append((buf, sorted(buf_pages)))   # 再放就超上限，先把已有内容成块
                buf, buf_pages = "", set()
            buf += text            # 直接拼接（段落已清理过空白）
            if page is not None:
                buf_pages.add(page)
            if len(buf) >= self.min_chunk_size:
                chunks.append((buf, sorted(buf_pages)))   # 达到下限即可成块
                buf, buf_pages = "", set()

        # 第 3 步：收尾，剩余的短内容若能装进最后一块（不超上限）就并入，否则独立成块
        if buf:
            if chunks and len(chunks[-1][0]) + len(buf) <= self.max_chunk_size:
                prev_text, prev_pages = chunks[-1]
                chunks[-1] = (prev_text + buf, sorted(set(prev_pages) | buf_pages))
            else:
                chunks.append((buf, sorted(buf_pages)))   # 不足下限的"孤尾"：宁短勿丢
        return chunks

    def _split_long_paragraph(self, text):
        """把超长段落按句子边界（。！？!?；;）切成不超过 max_chunk_size 的子段"""
        # 正则：连续的非句末标点字符 + 可选的句末标点，即一个"句子"
        sentences = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)

        # 单句超长（如没有标点的长串）先硬切成不超过上限的小段
        pieces = []
        for s in sentences:
            while len(s) > self.max_chunk_size:
                pieces.append(s[:self.max_chunk_size])
                s = s[self.max_chunk_size:]
            if s:
                pieces.append(s)

        # 再把句子贪心拼装成不超过上限的子段
        result, buf = [], ""
        for s in pieces:
            if buf and len(buf) + len(s) > self.max_chunk_size:
                result.append(buf)
                buf = s
            else:
                buf += s
        if buf:
            result.append(buf)
        return result

    # ---------- 第 4 步：知识点抽取（每 5 块合并一次请求，失败自动降级为单块） ----------
    def _extract_knowledge(self, chunks, progress_cb=None):
        """
        知识点抽取主流程：
          按 knowledge_batch_size（默认 5）个片段一组合并请求 -> 按顺序拆分结果 -> 汇总去重。
          某一批批量调用失败时，自动降级为该批逐块单独调用，保证整体流程不中断。
        :param progress_cb: 每批完成后触发 progress_cb(已完成批数, 总批数, 已累计候选数)
        """
        candidates, seen = [], set()
        n_batches = (len(chunks) + self.knowledge_batch_size - 1) // self.knowledge_batch_size
        done = 0
        for i in range(0, len(chunks), self.knowledge_batch_size):
            batch = chunks[i:i + self.knowledge_batch_size]
            names = self._extract_batch_with_fallback(batch)
            for name in names:                  # 按出现顺序去重
                name = str(name).strip()
                if name and name not in seen:
                    seen.add(name)
                    candidates.append(name)
            done += 1
            if progress_cb:
                progress_cb(done, n_batches, len(candidates))
        return candidates

    def _extract_batch_with_fallback(self, batch):
        """
        抽取一批片段的知识点，带降级保护：
          1) 优先走批量调用，返回后按顺序拆分成"每个片段一组"；
          2) 批量 API 失败 / 返回格式无法对应到各片段 / 组数与片段数不一致时，
             自动降级为该批逐块单独调用（宁慢勿错）。
        :return: 该批全部知识点名称列表（扁平）
        """
        per_chunk = self._ask_knowledge_batch(batch)

        if per_chunk is not None:
            # 形态一：模型返回扁平字符串数组（无法区分块，但候选本就是汇总去重的，直接采用）
            if per_chunk and all(isinstance(x, str) for x in per_chunk):
                self.logger.info("批量抽取成功（扁平式）：%d 个知识点", len(per_chunk))
                return per_chunk
            # 形态二：分组式——组数与片段数一致才视为拆分成功
            if len(per_chunk) == len(batch):
                flat = [n for names in per_chunk for n in names]
                self.logger.info("批量抽取成功（分组式）：%d 片段 -> %d 个知识点", len(batch), len(flat))
                return flat
            self.logger.warning("批量返回组数(%d)与片段数(%d)不一致", len(per_chunk), len(batch))

        # ---- 降级路径：逐块单独调用 ----
        self.logger.warning("批量抽取失败，降级为单块调用（%d 个片段）", len(batch))
        names = []
        for c in batch:
            names.extend(self._ask_knowledge_single(c["text"]))
        return names

    def _ask_knowledge_batch(self, batch):
        """
        单次 API 调用抽取一批片段的知识点。
        :return: 按顺序拆分后的结果——分组式 list[list[str]]、扁平式 list[str]；失败返回 None
        """
        segs = "\n\n".join(f"【片段{i + 1}｜{c['id']}】\n{c['text']}" for i, c in enumerate(batch))
        prompt = f"""请分别判断下面 {len(batch)} 个课件片段各自包含的知识点（如概念、原理、公式、定理、方法、事件等）。

{segs}

严格要求：
1. 对每个片段输出一个条目，条目顺序必须与片段顺序一致；
2. 只输出一个 JSON 数组，格式如下（knowledge_points 为该片段包含的知识点名称，每个不超过 20 字，没有则用空数组）：
[{{"chunk_id": "片段编号", "knowledge_points": ["知识点1", "知识点2"]}}]
3. 不要输出任何解释文字或代码块标记。"""
        data = self.chat_json(prompt, temperature=0.2)  # 抽取类任务用低温度
        return self._split_batch_result(data)

    @staticmethod
    def _split_batch_result(data):
        """
        按顺序拆分批量返回的 JSON 结果，兼容三种形态：
          1. 标准分组式：[{"chunk_id": "...", "knowledge_points": [...]}, ...]（按位置对应片段）
          2. 纯嵌套数组式：[["知识点A"], [], ["知识点B"], ...]
          3. 扁平汇总式：["知识点A", "知识点B", ...]（无法区分块，整体返回）
        解析失败（非数组/空数组/结构无效）返回 None，由调用方降级。
        """
        # 兼容模型把数组包在对象里的情况，如 {"results": [...]}
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), None)
        if not isinstance(data, list) or not data:
            return None                      # 空数组多半是没按格式返回，交给降级逻辑
        # 形态 3：扁平字符串数组，直接整体返回
        if all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x.strip()]
        # 形态 1 / 2：逐条拆分，未知条目用空列表占位以保持顺序对齐
        per_chunk = []
        for item in data:
            if isinstance(item, dict):
                names = item.get("knowledge_points", item.get("points", []))
                per_chunk.append([str(n).strip() for n in names if str(n).strip()]
                                 if isinstance(names, list) else [])
            elif isinstance(item, list):
                per_chunk.append([str(n).strip() for n in item if str(n).strip()])
            else:
                per_chunk.append([])
        return per_chunk

    def _ask_knowledge_single(self, text):
        """单块抽取（降级路径）：失败返回空列表，不影响其他片段"""
        prompt = f"""请判断下面的课件片段包含的知识点（如概念、原理、公式、定理、方法、事件等）。

课件片段：
{text}

要求：
1. 提取知识点名称，每个不超过 20 字；没有知识点则输出 []；
2. 只输出一个 JSON 字符串数组，例如 ["牛顿第二定律"]；
3. 不要输出任何解释文字或代码块标记。"""
        data = self.chat_json(prompt, temperature=0.2)
        if isinstance(data, dict):   # 兼容对象包装
            data = next((v for v in data.values() if isinstance(v, list)), [])
        return [str(n).strip() for n in data if str(n).strip()] if isinstance(data, list) else []


# 命令行快速测试：python agents/parser_agent.py <PDF路径>
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("用法：python agents/parser_agent.py <PDF路径>")
    else:
        result = ParserAgent().parse(sys.argv[1])
        preview = {**result, "chunks": result["chunks"][:3]}  # 只打印前几个块避免刷屏
        print(json.dumps(preview, ensure_ascii=False, indent=2))
