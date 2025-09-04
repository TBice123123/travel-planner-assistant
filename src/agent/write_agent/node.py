from typing import cast
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.agent.state import State
from src.agent.tools import write_note
from langchain_openai_like import init_openai_like_chat_model
from src.agent.write_agent.prompts import SUMMARY_PROMPT, WRITE_PROMPT
from langgraph.prebuilt import ToolNode


async def write(state: State):
    task_messages = state["task_messages"] if "task_messages" in state else []

    write_model = init_openai_like_chat_model(
        model="qwen-flash", provider="dashscope"
    ).bind_tools([write_note], tool_choice="write_note")

    chain = ChatPromptTemplate.from_template(WRITE_PROMPT) | write_model

    task_content = task_messages[-1].content

    # 使用create_task启动两个任务
    response = cast(AIMessage, await chain.ainvoke({"task_result": task_content}))

    return {
        "write_note_messages": [response],
    }


async def summary(state: State):
    task_messages = state["task_messages"] if "task_messages" in state else []
    summary_model = init_openai_like_chat_model(
        model="qwen-flash", provider="dashscope"
    )
    chain2 = ChatPromptTemplate.from_template(SUMMARY_PROMPT) | summary_model

    task_content = task_messages[-1].content
    response = cast(AIMessage, await chain2.ainvoke({"task_result": task_content}))
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
