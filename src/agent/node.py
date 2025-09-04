from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
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
from langchain_openai_like import init_openai_like_chat_model


async def call_model(state: State) -> Command[Literal["tools", "subagent", "__end__"]]:
    model = init_openai_like_chat_model(model="deepseek-chat", provider="deepseek")
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
