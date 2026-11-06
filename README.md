# AI PM 学习资料问答助手

基于官方 `rag-from-scratch` 的 Indexing → Retrieval → Generation 思路，用少量 Python 代码实现的最小 RAG 项目。不使用 LangChain、Chroma、Agent、GraphRAG、混合检索或 Rerank。除原 CLI 外，现提供 Flask + 原生 HTML/CSS/JavaScript 的轻量 Web UI，无需 Node.js。

知识库默认读取工作区中的：

- `AI产品经理能力全景指南.md`
- `秋招 AI 面试八股准备手册.md`

## 完整链路

```text
Markdown 加载
  → 按标题分节并在节内重叠切分（保留来源、标题路径）
  → OpenAI 兼容 Embedding API
  → NumPy 本地向量索引
  → 余弦相似度 Top-K 检索
  → 带资料编号的 Prompt
  → OpenAI 兼容 Chat API
  → 模型引用 + 程序提供的来源片段
```

## 安装

```bash
cd Day07_RAG/aipm_rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果 PyPI 下载超时，可以增加断点续传重试：

```bash
python -m pip install --resume-retries 10 --timeout 60 -r requirements.txt
```

## 配置模型

```bash
cp .env.example .env
```

项目默认使用硅基流动的免费模型组合：`BAAI/bge-m3` 负责 Embedding，`XingChenAGI/Xing4.0-29B` 负责回答。在硅基流动控制台创建 API Key，然后编辑本地 `.env`：

```dotenv
OPENAI_API_KEY=你的硅基流动APIKey
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_EMBEDDING_MODEL=BAAI/bge-m3
OPENAI_CHAT_MODEL=XingChenAGI/Xing4.0-29B
```

也可使用兼容 OpenAI API 的其他服务，但该服务必须同时支持所配置的 Embedding 和 Chat 模型。`.env` 已被 `.gitignore` 排除；不要打印、硬编码或提交真实密钥。免费模型的价格、可用性和限流策略以硅基流动控制台为准。

切分与检索参数也可在 `.env` 修改：

```dotenv
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
RAG_TOP_K=4
RAG_MIN_SCORE=0.45
RAG_REQUEST_TIMEOUT=120
```

`RAG_MIN_SCORE` 是最高检索相似度的拒答门槛，取值范围为 -1 到 1。低于门槛时直接回答“知识库中未找到足够依据”，不再调用回答模型，但仍展示实际检索记录。它与 Prompt 约束共同防止模型对低相关资料强行作答，可按自己的评测集调整。

默认文档路径和索引目录可通过 `RAG_DOCUMENT_PATHS`、`RAG_INDEX_DIR` 覆盖。多个文档路径用英文逗号分隔，相对路径以本项目目录为基准。

## 构建索引

```bash
python build_index.py
```

索引保存在 `.index/`：

- `vectors.npy`：归一化后的向量矩阵；
- `chunks.json`：chunk 正文、ID、来源文件和标题路径；
- `manifest.json`：文档 SHA-256、切分参数、Embedding 模型和服务地址。

再次运行时，如果文档和索引配置没有变化，会直接复用索引。文档或配置变化会自动重建；也可强制执行：

```bash
python build_index.py --force
```

## 命令行提问

```bash
python ask.py "AI 产品经理应该如何评估 RAG 系统？"
```

CLI 会先展示 Top-K 相似度、来源文件、标题和检索片段，再展示最终回答及来源附录。

只验证 Embedding 和检索，不调用回答模型：

```bash
python ask.py "RAG 检索质量应该看哪些指标？" --retrieve-only
```

临时修改 Top-K 或强制重建：

```bash
python ask.py "RAG 和微调怎么选？" --top-k 4
python ask.py "RAG 和微调怎么选？" --rebuild
```

建议分别测试库内与库外问题：

```bash
python ask.py "AI 产品经理应该如何评估 RAG 系统？"
python ask.py "如何做一道红烧肉？"
```

第二个问题应回答“知识库中未找到足够依据”。向量检索即使对库外问题也会返回相似度最高的片段，所以低相关结果不能当作已有依据；当前版本同时使用相似度门槛和 Prompt 拒答约束。

## Web 前端

```bash
source .venv/bin/activate
python web_app.py
```

打开 <http://127.0.0.1:8000>。服务默认只监听本机；可用 `--host` 和 `--port` 修改，例如 `python web_app.py --port 8080`。

前端支持：

- 连续提问、加载状态、失败重试、清空对话和复制回答；
- 调整 Top-K，查看每段资料的来源文件、标题路径、相似度和完整原文；
- 点击回答中的 `[1]` 等引用，展开并高亮对应资料；
- 手机端单列布局和检索资料抽屉；
- 通过 `http://127.0.0.1:8000/?q=你的问题` 打开页面并自动提问。

接口为 `GET /api/status` 和 `POST /api/ask`。后者接收 `{"question":"你的问题","top_k":4}`，返回模型回答、真实检索结果和耗时；不会在响应中返回 API Key。

## 代码导览

- `documents.py`：Markdown 加载、标题栈、递归切分、稳定 chunk ID。
- `embeddings.py`：把来源、标题和正文一起向量化，批量调用 Embedding API。
- `vector_store.py`：本地索引、余弦相似度 Top-K、内容指纹缓存。
- `rag.py`：约束 Prompt、检索上下文组装、Chat 调用、来源附录。
- `ask.py`：问答和检索 CLI。
- `build_index.py`：索引构建 CLI。
- `config.py`：环境变量、默认切分参数、相关度门槛和请求超时。
- `web_app.py`：Flask 页面、状态与问答接口，默认仅监听本机。
- `web/`：原生 HTML/CSS/JavaScript 前端，无外部 CDN 或前端构建步骤。

## 前端设计 skill

已参考 [Leonxlnx/taste-skill](https://github.com/Leonxlnx/taste-skill) 仓库中 `skills/taste-skill/SKILL.md` 的设计规范，并安装为用户级技能，后续项目可复用：

```text
~/.kimi-code/skills/design-taste-frontend/SKILL.md
~/.kimi-code/skills/design-taste-frontend/LICENSE
```

本机没有 Node.js，因此只采用了技能中的设计规范，未运行其需要 Node 22+ 的官方检查脚本；页面改用真实 Chrome 浏览器验收。前端实现本身不需要 Node.js。

## 验证记录

离线验证：

- 两份文档成功加载并切成 152 个 chunk，分别为 84 和 68 个；最大 chunk 长度为 794。
- 所有 chunk 都有来源、标题路径和唯一 ID。
- 用已知向量验证了本地保存/加载、余弦相似度排序和 Top-K。
- 验证了 Prompt 的必要约束、上下文注入和来源附录格式。
- 未配置模型时，构建和问答命令会明确提示缺少 `OPENAI_API_KEY`，不会返回模拟答案。

使用硅基流动 `BAAI/bge-m3 + XingChenAGI/Xing4.0-29B` 的真实验收：

- 成功为 152 个 chunk 建立 1024 维向量索引，后续问答确认复用索引，没有重复建库。
- 库内问题“AI 产品经理应该如何评估 RAG 系统？”命中评测与 RAG 章节，回答包含 Recall@K、MRR、忠实度、答案相关性、完整性和引用准确性，并附真实来源。
- 库外问题“如何做一道红烧肉？”的最高相似度仅约 0.41，最终只回答“知识库中未找到足够依据”，没有编造做法。
- 免费模型可能出现较长排队、超时或服务波动，也曾出现无视库外问题而总结资料的情况；因此增加默认 0.45 的相关度门槛。Embedding 和 Chat 默认单请求超时为 120 秒，可通过 `RAG_REQUEST_TIMEOUT` 调整。

Web 验收：

- `pip check`、页面与静态资源、状态接口、空问题和非法 Top-K 校验通过，原 CLI 真实问答未受影响。
- Web API 的库内回答、4 条真实来源和库外拒答均已验证；模型超时返回 504，服务错误返回 502，不使用模拟答案。
- 在真实 Chrome 中完成桌面问答、引用点击展开并高亮、清空对话，以及 390px 设备仿真下的拒答和资料抽屉验证；页面无横向溢出。
