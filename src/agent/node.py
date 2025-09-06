from typing import Literal, cast

from langchain_core.messages import AIMessage, SystemMessage
from langchain_dev_utils import has_tool_calling, load_chat_model, parse_tool_calling
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime
from langgraph.types import Command

from src.agent.state import State
from src.agent.tools import (
    ls,
    query_note,
    transfor_task_to_subagent,
    update_todo,
    write_todo,
)
from src.agent.utils.context import Context


async def call_model(state: State) -> Command[Literal["tools", "subagent", "__end__"]]:
    run_time = get_runtime(Context)
    model = load_chat_model(
        model=run_time.context.todo_model,
    )

    tools = [
        write_todo,
        update_todo,
        transfor_task_to_subagent,
        ls,
        query_note,
    ]
    bind_model = model.bind_tools(tools, parallel_tool_calls=False)
    messages = state["messages"]

    response = await bind_model.ainvoke(
        [SystemMessage(content=run_time.context.todo_prompt), *messages]
    )

    if has_tool_calling(cast(AIMessage, response)):
        name, _ = parse_tool_calling(
            cast(AIMessage, response), first_tool_call_only=True
        )
        if name == "transfor_task_to_subagent":
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
