# 智学 AI 学习助手（agents 架构版）

基于 Streamlit + DeepSeek API 的 AI 学习助手。采用多智能体（Multi-Agent）架构：每个智能能力独立为一个 Agent 模块，app.py 只负责页面交互与流程编排。

## 功能闭环（7 个页面）

1. 📤 **上传解析**：PDF 提取文本 → 按 300-800 字分块 → 抽取知识点 → 建立检索索引 → 自动构建知识图谱
2. 💬 **智能问答**：混合检索（向量召回 Top-5 + 知识图谱 2 跳扩展，重排序 0.7/0.3）→ 基于课件回答问题 → 标注来源片段与命中知识点
3. 🕸️ **知识图谱**：AI 补全知识点描述与难度（基础/中等/困难），分析先修依赖，networkx 建图 + pyvis 可视化（节点颜色=难度、大小=度数，箭头=先修方向）
4. 📝 **学习诊断**：按图谱度数选核心知识点出题 → 作答 → 错因诊断报告（知识域/错误类型/薄弱点重复度）
5. 🗺️ **路径推荐**：结合诊断报告与图谱前置依赖，拓扑排序生成学习路径
6. 📊 **学习评估**：双模式——⚡快速评估（手输前后测分数算 ALG）/ 📝完整评估（前后测作答 + AI 增益报告）
7. 📚 **历史记录**：从 SQLite 读取全部学习记录，按类型筛选，展开回溯问答/诊断/评估完整内容

所有操作自动存入 SQLite，侧边栏可快速查看最近 8 条记录。

## 架构

```
app.py                    Streamlit 主程序（页面 + 流程编排）
agents/
  base_agent.py           公共基类：DeepSeek 客户端、chat/chat_json、JSON 稳健解析
  parser_agent.py         ParserAgent     PDF -> 文本块 + 知识点候选
  ontology_agent.py       OntologyAgent   知识图谱构建（nodes/edges 校验清洗）
  index_agent.py          IndexAgent      本地 TF-IDF 检索索引（bigram + numpy）
  retriever_agent.py      RetrieverAgent  混合检索：向量召回 + 图谱 2 跳扩展 + 重排序
  tutor_agent.py          TutorAgent      引导式答疑（先引导思考，标注来源与知识点）
  diagnosis_agent.py      DiagnosisAgent  图谱选点出题 + 错因诊断（JSON 报告，含薄弱点重复度）
  path_agent.py           PathAgent       学习路径（拓扑排序：先修在前、薄弱在后，有环用近似序）
  eval_agent.py           EvalAgent       前测-后测-学习增益评估（ALG 公式，报告入库）
utils/
  db.py                   SQLite 数据访问层（files/qa/diagnosis/path/eval 五表 + 增查函数）
  graph_viz.py            知识图谱可视化（networkx -> pyvis HTML 字符串，难度配色）
requirements.txt          依赖清单
.env                      DEEPSEEK_API_KEY（不要提交到 Git）
learning_records.db       运行后自动生成的 SQLite 数据库
```

## 运行步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥（项目根目录 .env 文件）
DEEPSEEK_API_KEY=你的DeepSeek密钥

# 3. 启动应用
streamlit run app.py
```

## 技术栈

- 前端：Streamlit（图谱用 pyvis/vis.js 渲染）
- 大模型：DeepSeek API（OpenAI SDK 兼容调用）
- 检索：向量召回（IndexAgent，numpy 余弦相似度）+ 图谱扩展（networkx 2 跳子图）+ 加权重排序（0.7 向量 + 0.3 图谱）
- 图谱数据结构：networkx
- PDF 解析：pdfplumber
- 数据库：SQLite3
