# Shuiyuan Agent

一个面向上海交通大学校园问题的只读 Shuiyuan 社区搜索 Agent。

系统使用校内部署的 LLM 理解问题、生成适合 Shuiyuan 子串搜索的短查询、判断候选帖价值、按需展开帖子回复，并生成带可跳转引用的回答。

当前版本已经支持本地会话记忆、多轮追问和聊天式 Web 界面，适合连续追问同一校园主题，而不是一次性单轮搜索。

> 本项目不是 Shuiyuan 社区或上海交通大学的官方服务。请遵守社区规则、授权范围与访问频率限制。

## 工作流程

```text
用户问题
  -> LLM 生成搜索意图、桥接概念和多样化短查询
  -> 使用 Discourse User API Key 低频搜索 Shuiyuan
  -> LLM 判断证据相关性、适用范围和时效性
  -> 按需读取高价值帖子正文
  -> 从正文片段中整理事实并生成回答
  -> 将当前轮摘要、对象集合和上下文写入本地会话记忆
  -> 用户可继续追问，LLM 结合上下文做 follow-up 解析
```

## 核心能力

- LLM 动态搜索规划，而不是固定关键词改写
- 对校园部门、地点、简称和社区表达进行桥接推理
- 按需阅读帖子回复并提取相关正文片段
- 区分直接答案、补充信息、背景和无关内容
- 处理近期帖子、综合帖、Wiki 与具体案例之间的关系
- 支持多轮对话、历史会话切换和会话内追问
- 对追问问题解析当前话题、对象集合和缺失属性
- 在列举类问题中归并候选对象，而不是只按帖子顺序输出
- 只读访问 Shuiyuan，并对限流响应等待后继续
- 对超时请求做有限次等待后重试
- Web UI 展示搜索计划、运行阶段和采用的证据
- Web UI 提供全局对话偏好、历史对话列表和聊天式消息流

## 安全边界

- LLM 默认只允许访问 `https://models.sjtu.edu.cn/api/v1`
- Shuiyuan 默认只允许访问 `https://shuiyuan.sjtu.edu.cn`
- Web 授权固定申请 Discourse `read` scope
- 模型 API Key 默认不持久化；Shuiyuan 只读 User-Api-Key 会保存在浏览器本地存储，便于后续复用
- Shuiyuan 帖子按需读取，不持久化保存

详见 [SECURITY.md](SECURITY.md)。

## 快速开始

```bash
git clone https://github.com/Carl1707/shuiyuan-agent.git
cd shuiyuan-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env.local
```

在 `.env.local` 中填写校内模型 API Key：

```dotenv
SJTU_LLM_API_KEY=your-sjtu-model-api-key
SJTU_LLM_API_BASE=https://models.sjtu.edu.cn/api/v1
SJTU_LLM_MODEL=deepseek-chat
```

`SJTU_LLM_MODEL` 可以填写获授权的校内模型调用名，例如
`deepseek-chat`、`deepseek-reasoner`、`minimax`、`glm` 或 `qwen`。

启动：

```bash
campus-agent --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000`：

1. 确认校内模型已配置，或在页面临时填写模型 API Key
2. 点击“开始只读授权”，在 Shuiyuan 确认授权
3. 将返回的加密 payload 粘贴回页面并完成授权
4. 输入校园问题并查看搜索与证据整理过程
5. 如需连续追问，可直接在同一会话中继续提问；历史会话会保存在本地 SQLite 中

## 项目结构

```text
campus_agent/
  agent.py       搜索规划、正文读取、证据整理与回答流程
  llm.py         校内模型调用和结构化推理
  session_store.py 本地会话记忆、历史对话与上下文压缩
  tools.py       Shuiyuan Discourse 只读 API 客户端
  web.py         本地 Web UI 与只读授权流程
  retrieval.py   当前请求内的帖子正文片段排序
  chunking.py    当前请求内的帖子正文切片
```

## License

[MIT](LICENSE)
