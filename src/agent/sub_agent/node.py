from typing import Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_dev_utils import (
    has_tool_calling,
    load_chat_model,
    message_format,
    parse_tool_calling,
)
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime
from langgraph.types import Command

from src.agent.sub_agent.state import SubAgentState
from src.agent.tools import (
    get_weather,
    query_note,
    tavily_search,
    write_note,
)
from src.agent.utils.context import Context


async def subagent_call_model(
    state: SubAgentState,
) -> Command[Literal["sub_tools", "write_and_summary", "__end__"]]:
    run_time = get_runtime(Context)  # 获取运行时环境，也就是上下文配置
    if isinstance(state["messages"][-1], AIMessage):
        last_ai_message = state["messages"][-1]
    else:
        last_ai_message = cast(AIMessage, state["messages"][-2])

    _, args = parse_tool_calling(last_ai_message, first_tool_call_only=True) # 获取工具调用的参数
    task_name = cast(dict, args).get("content", "") # 获取待办事项名称

    model = load_chat_model(model=run_time.context.sub_model).bind_tools(
        [get_weather, tavily_search, query_note, write_note]
    )  # 构建子agent模型

    messages = state.get("temp_task_messages", []) # 获取临时任务消息

    notes = state.get("note", {}) # 获取笔记

    user_requirement = state["messages"][0].content # 获取用户需求

    response = await model.ainvoke(
        [
            SystemMessage(
                content=run_time.context.sub_prompt.format(
                    task_name=task_name,
                    history_files=message_format(list(notes.keys()))
                    if notes
                    else "当前没有笔记",
                    user_requirement=user_requirement,
                )
            ),
            HumanMessage(content=f"我的任务是：{task_name}，请帮我完成"),
            *messages,
        ]
    )

    if has_tool_calling(cast(AIMessage, response)):  # 如果有工具调用
        name, _ = parse_tool_calling(
            cast(AIMessage, response), first_tool_call_only=True
        )
        if name == "write_note":
            return Command(
                goto="write_and_summary",
                update={"temp_task_messages": [response]},
            )
        else:
            return Command(
                goto="sub_tools",
                update={"temp_task_messages": [response]},
            )

    return Command(
        goto="__end__",
        update={
            "task_messages": [*messages, response],
        },
    )  # 任务结束，将response添加到task_messages中，也就是主agent的messages中


sub_tools = ToolNode(
    [get_weather, tavily_search, query_note], messages_key="temp_task_messages"
) # messages_key 指定从subagent的temp_task_messages中获取消息
