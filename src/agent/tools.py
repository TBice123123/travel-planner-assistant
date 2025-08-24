from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.agent.state import State, Todo


@tool
def write_todo(todos: list[str], tool_call_id: Annotated[str, InjectedToolCallId]):
    """用于写入todo的工具,只能使用一次，在最开始的时候使用，后续请使用update_todo更新。
    参数：
    todos: list[str], 待写入的todo列表，这是一个字符串列表，每个字符串都是一个todo内容content
    """

    return Command(
        update={
            "todo": [
                {"content": todo, "status": "pending" if index > 0 else "in_progress"}
                for index, todo in enumerate(todos)
            ],
            "messages": [
                ToolMessage(
                    content=f"Todo list 写入成功，下面请先执行{todos[0]}任务（无需修改状态为in_process）",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def update_todo(
    update_todos: list[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
):
    """用于更新todo的工具, 可以多次使用，每次使用都会更新todo列表。
    参数：
    update_todos: list[Todo], 更新的todo列表，这是一个字典列表,对于该字典的内容如下：
    每个字典包含两个字段：
    content: str, todo内容
    status: str, todo状态, 可选值有pending, in_progress, done，但是你传入的status的值只有可能是in_progress或者done

    更新的时候，只需要传入要更新的todo列表即可，无需传入所有的todo列表。

    注意：传入的update_todos中，status为in_progress和done的todo必须同时出现，即至少有一个status为in_progress的todo和至少有一个status为done的todo。

    例如当前的todo有：
    1. 待办1
    2. 待办2
    3. 待办3
    4. 待办4
    5. 待办5
    此时，完成了待办1，接下来需要完成待办2,则输入的todo_list应该为：
    [
        {"content":"待办1"，
            "status":"done"
        },
        {"content":"待办2"，
            "status":"in_progress"
        }
    ]

    """

    todo_list = state["todo"] if "todo" in state else []

    updated_todo_list = []

    # 遍历update_todo,每遍历一个todo，就从todo_list中查找对应的todo，并更新其状态
    for update_todo in update_todos:
        for todo in todo_list:
            if todo["content"] == update_todo["content"]:
                todo["status"] = update_todo["status"]
                updated_todo_list.append(todo)

    # 检查是否所有的todo were updated
    if len(updated_todo_list) < len(update_todos):
        raise ValueError(
            "未找到如下的todo:"
            + ",".join(
                [
                    todo["content"]
                    for todo in update_todos
                    if todo not in updated_todo_list
                ]
            )
            + "请检查todo列表是否正确，目前的todo列表为:"
            + "\n".join(
                [todo["content"] for todo in todo_list if todo["status"] != "done"]
            )
        )

    return Command(
        update={
            "todo": todo_list,
            "messages": [
                ToolMessage(content="Todo list 更新成功", tool_call_id=tool_call_id)
            ],
        }
    )


@tool
async def transfor_task_to_subagent(
    content: Annotated[
        str, "当前待执行的todo任务内容，必须与todo列表中待办事项的content字段完全一致"
    ],
):
    """用于执行todo任务的工具。

    参数：
    content: str, 待执行的todo任务内容，必须与todo列表中待办事项的content字段完全一致

    例如当前的todo list是
    [
        {"content":"待办1"，"status":"done"}
        {"content":"待办2"，"status":"in_progress"}
        {"content":"待办3"，"status":"pending"}

    ]
    则可以知道当前执行的是待办2，则输入的content应该为"待办2"。
    """

    return "transfor success!"


@tool
def write_note(
    file_name: Annotated[str, "笔记的名称"],
    content: Annotated[str, "笔记的内容"],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """用于写入笔记的工具。

    参数：
    content: str, 笔记内容

    """

    return Command(
        update={
            "note": {file_name: content},
            "write_note_messages": [
                ToolMessage(
                    content=f"笔记{file_name}写入成功，内容是{content}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def ls(state: Annotated[State, InjectedState]):
    """列出笔记列表。

    返回：
    list[str], 笔记列表

    """
    notes = state["note"] if "note" in state else {}
    return list(notes.keys())


@tool
def query_note(file_name: str, state: Annotated[State, InjectedState]):
    """查询笔记。

    参数：
    file_name:笔记名称

    返回：
    str, 查询的笔记内容

    """
    notes = state["note"] if "note" in state else {}

    return notes.get(file_name, "未找到笔记名称")


@tool
def get_weather(city: str):
    """查询天气。

    参数：
    city:城市名称

    返回：
    str, 天气信息

    """
    return f"{city}的天气是晴天，温度是25度。"
