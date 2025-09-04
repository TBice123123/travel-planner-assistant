from typing import Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from src.agent.state import State
from src.agent.sub_agent.prompts import SUBAGENT_PROMPT
from src.agent.tools import get_weather, query_note, tavily_search
from src.agent.utils.model import load_chat_model


async def subagent_call_model(state: State) -> Command[Literal["sub_tools", "__end__"]]:
    last_ai_message = cast(AIMessage, state["messages"][-1])

    task_name = last_ai_message.tool_calls[0]["args"].get("content", "")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUBAGENT_PROMPT),
            ("human", "我的任务是：{task_name}，请帮我完成"),
            ("placeholder", "{placeholder}"),
        ]
    )

    # model = load_chat_model(
    #     model_name="qwen3-235b-a22b-instruct-2507", model_provider="dashscope"
    # ).bind_tools([get_weather, tavily_search, query_note])

    model = load_chat_model(model_name="glm-4.5", model_provider="zai").bind_tools(
        [get_weather, tavily_search, query_note]
    )

    chain = prompt | model
    task_messages = state["task_messages"] if "task_messages" in state else []

    now_task_message_index = (
        state["now_task_message_index"] if "now_task_message_index" in state else 0
    )

    messages = task_messages[now_task_message_index:]

    notes = state["note"] if "note" in state else {}

    response = await chain.ainvoke(
        {
            "placeholder": messages,
            "task_name": task_name,
            "history_files": "\n".join([f"- {note_name}" for note_name in notes.keys()])
            or "暂无历史记录文件",
            "user_requirement": state["messages"][0].content,
        }
    )

    if (
        isinstance(response, AIMessage)
        and hasattr(response, "tool_calls")
        and len(response.tool_calls) > 0
    ):
        return Command(
            goto="sub_tools",
            update={"task_messages": [response]},
        )

    return Command(
        goto="__end__",
        update={
            "task_messages": [response],
        },
    )


sub_tools = ToolNode(
    [get_weather, tavily_search, query_note], messages_key="task_messages"
)
