# deep-agent-langgraph

该项目基于 langchain 官方的`deepagents`进行构建，但加入了许多个人的理解和改进，并适配了国内的两款顶尖模型`deepseek`和`qwen3`模型。

## 项目本地配置方式

首先克隆本仓库

```bash
git clone git@github.com:TBice123123/deep-agent-langgraph.git
```

进入项目目录

```bash
cd deep-agent-langgraph
```

安装项目依赖

```bash
uv sync
```

配置环境变量

```bash
cp .env.example .env
```

需要配置的环境变量有：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_API_BASE`
- `DEEPSEEK_API_KEY`

启动项目

```bash
uv run langgraph dev --host=localhost
```
