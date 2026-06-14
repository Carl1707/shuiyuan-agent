# Shuiyuan Agent

一个面向上海交通大学校园问题的只读 Shuiyuan 社区搜索 Agent。

系统使用校内部署的 LLM 理解问题、生成适合 Shuiyuan 子串搜索的短查询、判断候选帖价值、按需展开帖子回复，并生成带可跳转引用的回答。

> 本项目不是 Shuiyuan 社区或上海交通大学的官方服务。请遵守社区规则、授权范围与访问频率限制。

## 工作流程

```text
用户问题
  -> LLM 生成搜索意图、桥接概念和多样化短查询
  -> 使用 Discourse User API Key 低频搜索 Shuiyuan
  -> LLM 判断证据相关性、适用范围和时效性
  -> 按需读取高价值帖子正文
  -> 从正文片段中整理事实并生成回答
```

## 核心能力

- LLM 动态搜索规划，而不是固定关键词改写
- 对校园部门、地点、简称和社区表达进行桥接推理
- 按需阅读帖子回复并提取相关正文片段
- 区分直接答案、补充信息、背景和无关内容
- 处理近期帖子、综合帖、Wiki 与具体案例之间的关系
- 只读访问 Shuiyuan，并对限流响应等待后继续
- Web UI 展示搜索计划、运行阶段和采用的证据

## 安全边界

- LLM 默认只允许访问 `https://models.sjtu.edu.cn/api/v1`
- Shuiyuan 默认只允许访问 `https://shuiyuan.sjtu.edu.cn`
- Web 授权固定申请 Discourse `read` scope
- 页面不会将 API Key 写入浏览器存储
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

## 项目结构

```text
campus_agent/
  agent.py       搜索规划、正文读取、证据整理与回答流程
  llm.py         校内模型调用和结构化推理
  tools.py       Shuiyuan Discourse 只读 API 客户端
  web.py         本地 Web UI 与只读授权流程
  retrieval.py   当前请求内的帖子正文片段排序
  chunking.py    当前请求内的帖子正文切片
```

## License

[MIT](LICENSE)
