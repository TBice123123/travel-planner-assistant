import asyncio
from typing import cast
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.agent.state import State
from src.agent.tools import write_note
from src.agent.utils import load_chat_model
from src.agent.write_agent.prompts import SUMMARY_PROMPT, WRITE_PROMPT
from langgraph.prebuilt import ToolNode


async def write(state: State):
    task_messages = state["task_messages"] if "task_messages" in state else []

    write_model = load_chat_model(
        model_name="qwen-flash", model_provider="dashscope"
    ).bind_tools([write_note], tool_choice="write_note")

    summary_model = load_chat_model(model_name="qwen-flash", model_provider="dashscope")

    chain1 = ChatPromptTemplate.from_template(WRITE_PROMPT) | write_model
    chain2 = ChatPromptTemplate.from_template(SUMMARY_PROMPT) | summary_model

    task_content = task_messages[-1].content

    # 使用create_task启动两个任务
    coro1 = chain1.ainvoke({"task_result": task_content})
    coro2 = chain2.ainvoke({"task_result": task_content})
    task1 = asyncio.create_task(coro1)
    task2 = asyncio.create_task(coro2)
    
    # 等待两个并发任务完成
    result1 = await task1
    result2 = await task2

    response = cast(AIMessage, result1)
    summary = cast(AIMessage, result2)

    file_name = response.tool_calls[0]["args"]["file_name"]

    tool_call_id = cast(AIMessage, state["messages"][-1]).tool_calls[0]["id"]

    return {
        "write_note_messages": [response],
        "messages": [
            ToolMessage(
                content=f"任务执行完成！任务保存的文件名是：{file_name}\n 任务结果摘要： {summary.content}",
                tool_call_id=tool_call_id,
            ),
        ],
    }


write_tool = ToolNode([write_note], messages_key="write_note_messages")
