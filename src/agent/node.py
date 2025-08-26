from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from src.agent.prompts import TODO_MODEL_PROMPT
from src.agent.state import State
from src.agent.tools import (
    ls,
    query_note,
    transfor_task_to_subagent,
    update_todo,
    write_todo,
)


async def call_model(state: State) -> Command[Literal["tools", "subagent", "__end__"]]:
    model = ChatDeepSeek(model="deepseek-chat")
    # model = ChatDeepSeek(model="deepseek-ai/DeepSeek-V3.1") 如果使用硅基流动的模型请使用这个模型名称，同时设置DEEPSEEK_API_KEY和DEEPSEEK_API_BASE两个环境变量
    # model = ChatQwen(model="qwen3-235b-a22b-instruct-2507")

    tools = [
        write_todo,
        update_todo,
        transfor_task_to_subagent,
        ls,
        query_note,
    ]
    bind_model = model.bind_tools(tools, parallel_tool_calls=False)
    messages = state["messages"]

    template = ChatPromptTemplate.from_messages(
        [
            ("system", TODO_MODEL_PROMPT),
            ("placeholder", "{message_placeholder}"),
        ]
    )

    chain = template | bind_model

    response = await chain.ainvoke({"message_placeholder": messages})

    if (
        isinstance(response, AIMessage)
        and hasattr(response, "tool_calls")
        and len(response.tool_calls) > 0
    ):
        tool_call_name = response.tool_calls[0]["name"]
        if tool_call_name == "transfor_task_to_subagent":
            return Command(
                goto="subagent",
                update={
                    "messages": [response],
                    "now_task_message_index": len(
                        state["task_messages"] if "task_messages" in state else []
                    ),
                },
            )

        return Command(goto="tools", update={"messages": [response]})

    return Command(goto="__end__", update={"messages": [response]})


tool_node = ToolNode([write_todo, update_todo, ls, query_note])
