from typing import Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from src.agent.state import State

from src.agent.tools import get_weather, query_note, tavily_search
from langchain_dev_utils import has_tool_calling, load_chat_model
from langgraph.runtime import get_runtime
from src.agent.utils.context import Context


async def subagent_call_model(state: State) -> Command[Literal["sub_tools", "__end__"]]:
    run_time = get_runtime(Context)
    last_ai_message = cast(AIMessage, state["messages"][-1])

    task_name = last_ai_message.tool_calls[0]["args"].get("content", "")

    model = load_chat_model(model=run_time.context.sub_model).bind_tools(
        [get_weather, tavily_search, query_note]
    )

    task_messages = state["task_messages"] if "task_messages" in state else []

    now_task_message_index = (
        state["now_task_message_index"] if "now_task_message_index" in state else 0
    )

    messages = task_messages[now_task_message_index:]

    notes = state["note"] if "note" in state else {}

    response = model.invoke(
        [
            SystemMessage(
                content=run_time.context.sub_prompt.format(
                    task_name=task_name,
                    history_files="\n".join(
                        [f"- {note_name}" for note_name in notes.keys()]
                    )
                    or "暂无历史记录文件",
                    user_requirement=state["messages"][0].content,
                )
            ),
            HumanMessage(content=f"我的任务是：{task_name}，请帮我完成"),
            *messages,
        ]
    )

    if has_tool_calling(cast(AIMessage, response)):
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
