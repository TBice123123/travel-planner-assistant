from typing import cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_dev_utils import load_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime

from src.agent.write_agent.state import WriteState
from src.agent.tools import write_note
from src.agent.utils.context import Context


async def write(state: WriteState):
    run_time = get_runtime(Context)
    task_messages = state["task_messages"] if "task_messages" in state else []

    write_model = load_chat_model(
        model=run_time.context.write_model,
    ).bind_tools([write_note], tool_choice="write_note")

    task_content = task_messages[-1].content

    response = cast(
        AIMessage,
        await write_model.ainvoke(
            run_time.context.write_prompt.format(task_result=task_content)
        ),
    )

    return {
        "temp_write_note_messages": [response],
    }


async def summary(state: WriteState):
    run_time = get_runtime(Context)
    task_messages = state["task_messages"] if "task_messages" in state else []
    summary_model = load_chat_model(model=run_time.context.summary_model)

    task_content = task_messages[-1].content
    response = cast(
        AIMessage,
        await summary_model.ainvoke(
            run_time.context.summary_prompt.format(task_result=task_content)
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


write_tool = ToolNode([write_note], messages_key="temp_write_note_messages")
