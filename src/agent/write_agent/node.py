from typing import cast
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from src.agent.state import State
from src.agent.tools import write_note
from src.agent.write_agent.prompts import WRITE_PROMPT
from langgraph.prebuilt import ToolNode


async def write(state: State):
    task_messages = state["task_messages"] if "task_messages" in state else []

    model = ChatQwen(model="qwen-flash").bind_tools(
        [write_note], tool_choice="write_note"
    )

    chain = ChatPromptTemplate.from_template(WRITE_PROMPT) | model

    response = cast(
        AIMessage, await chain.ainvoke({"task_result": task_messages[-1].content})
    )

    file_name = response.tool_calls[0]["args"]["file_name"]

    tool_call_id = cast(AIMessage, state["messages"][-1]).tool_calls[0]["id"]

    return {
        "write_note_messages": [response],
        "messages": [
            ToolMessage(
                content=f"任务执行完成！任务保存的文件名是：{file_name}",
                tool_call_id=tool_call_id,
            ),
        ],
    }


write_tool = ToolNode([write_note], messages_key="write_note_messages")
