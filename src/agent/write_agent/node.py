from typing import cast
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from src.agent.state import State
from src.agent.tools import write_note
from langchain_dev_utils import load_chat_model
from langgraph.prebuilt import ToolNode
from src.agent.utils.context import Context
from langgraph.runtime import get_runtime


async def write(state: State):
    run_time = get_runtime(Context)
    task_messages = state["task_messages"] if "task_messages" in state else []

    write_model = load_chat_model(
        model=run_time.context.note_model,
    ).bind_tools([write_note], tool_choice="write_note")

    task_content = task_messages[-1].content

    response = cast(
        AIMessage,
        await write_model.ainvoke(
            [
                SystemMessage(
                    content=run_time.context.note_prompt.format(
                        task_result=task_content
                    )
                ),
                *task_messages,
            ]
        ),
    )

    return {
        "write_note_messages": [response],
    }


async def summary(state: State):
    run_time = get_runtime(Context)
    task_messages = state["task_messages"] if "task_messages" in state else []
    summary_model = load_chat_model(model=run_time.context.summary_model)

    task_content = task_messages[-1].content
    response = cast(
        AIMessage,
        await summary_model.ainvoke(
            [
                SystemMessage(
                    content=run_time.context.summary_prompt.format(
                        task_result=task_content
                    )
                ),
                *task_messages,
            ]
        ),
    )
    tool_call_id = cast(AIMessage, state["messages"][-1]).tool_calls[0]["id"]
    return {
        "messages": [
            ToolMessage(
                content=f"任务执行完成！此任务结果摘要：{response.content}",
                tool_call_id=tool_call_id,
            ),
        ],
    }


write_tool = ToolNode([write_note], messages_key="write_note_messages")
