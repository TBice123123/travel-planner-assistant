# Travel-planner-assistant

该项目基于 langchain 官方的`deepagents`进行构建，但加入了许多个人的理解和改进，并适配了国内的多款顶尖模型。

## 项目本地配置方式

首先克隆本仓库

```bash
git clone https://github.com/TBice123123/Travel-planner-Assistant.git

```

进入项目目录

```bash
cd travel-planner-assistant
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

- `TAVILY_API_KEY`
- 大模型 API Key（选其中几个）：
  - `DEEPSEEK_API_KEY`
  - `MOONSHOT_API_KEY`
  - `SILICONFLOW_API_KEY`
  - `DASHSCOPE_API_KEY`

启动项目

```bash
uv run langgraph dev --host=localhost --allow-blocking
```
