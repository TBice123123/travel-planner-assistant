from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode
from src.agent.sub_agent.state import SubAgentState
from src.agent.tools import write_note
from src.agent.utils.context import Context
from langgraph.runtime import get_runtime
from langchain_dev_utils import load_chat_model, parse_tool_calling
from langchain_core.messages import AIMessage, ToolMessage
from typing import cast

write_tool = ToolNode([write_note], messages_key="temp_task_messages")


async def summary(state: SubAgentState):
    run_time = get_runtime(Context)
    task_messages = state["temp_task_messages"] if "temp_task_messages" in state else []
    summary_model = load_chat_model(model=run_time.context.summary_model)

    _, args = parse_tool_calling(
        cast(AIMessage, task_messages[-1]), first_tool_call_only=True
    )
    task_content = cast(dict, args).get("content", "")
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
                content=response.content,
                tool_call_id=tool_call_id,
            )
        ],
    }


def build_write_and_summary_node():
    graph = StateGraph(SubAgentState)
    graph.add_node("write", write_tool)
    graph.add_node("summary", summary)
    graph.add_edge("__start__", "write")
    graph.add_edge("__start__", "summary")

    graph.add_edge("write", "__end__")
    graph.add_edge("summary", "__end__")
    return graph.compile()
