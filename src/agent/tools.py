from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_tavily.tavily_search import TavilySearch
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.agent.state import State
from langchain_dev_utils import (
    create_write_plan_tool,
    create_update_plan_tool,
    create_ls_tool,
    create_query_note_tool,
)


write_plan = create_write_plan_tool(
    name="write_plan",
    description="""用于写入计划的工具,只能使用一次，在最开始的时候使用，后续请使用update_plan更新。
参数：
plan: list[str], 待写入的计划列表，这是一个字符串列表，每个字符串都是一个计划内容content
""",
)

update_plan = create_update_plan_tool(
    name="update_plan",
    description="""用于更新计划的工具，可以多次使用来更新计划进度。
    参数：
    update_plans: list[Todo] - 需要更新的计划列表，每个元素是一个包含以下字段的字典：
        - content: str, 计划内容，必须与现有计划内容完全一致
        - status: str, 计划状态，只能是"in_progress"（进行中）或"done"（已完成）
    
    使用说明：
    1. 每次调用只需传入需要更新状态的计划，无需传入所有计划
    2. 必须同时包含至少一个"done"状态的计划和至少一个"in_progress"状态的计划
        - 将已完成的计划设置为"done"
        - 将接下来要执行的计划设置为"in_progress"
    3. content字段必须与现有计划内容精确匹配
    
    示例：
    假设当前计划列表为：
    [
        {"content":"计划1"，"status":"done"}
        {"content":"计划2"，"status":"in_progress"}
        {"content":"计划3"，"status":"pending"}
    ]
    当完成"计划1"并准备开始"计划2"时，应传入：
    [
        {"content":"计划1", "status":"done"},
        {"content":"计划2", "status":"in_progress"}
    ]
    """,
)

ls = create_ls_tool(
    name="ls",
    description="""用于列出所有已保存的笔记名称。

    返回：
    list[str]: 包含所有笔记文件名的列表

    """,
)

query_note = create_query_note_tool(
    name="query_note",
    description="""用于查询笔记。

    参数：
    file_name:笔记名称

    返回：
    str, 查询的笔记内容

    """,
)


# @tool
# def write_todo(todos: list[str], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """用于写入todo的工具,只能使用一次，在最开始的时候使用，后续请使用update_todo更新。
#     参数：
#     todos: list[str], 待写入的todo列表，这是一个字符串列表，每个字符串都是一个todo内容content
#     """

#     return Command(
#         update={
#             "todo": [
#                 {"content": todo, "status": "pending" if index > 0 else "in_progress"}
#                 for index, todo in enumerate(todos)
#             ],
#             "messages": [
#                 ToolMessage(
#                     content=f"Todo list 写入成功，下面请先执行{todos[0]}任务（无需修改状态为in_process）",
#                     tool_call_id=tool_call_id,
#                 )
#             ],
#         }
#     )


# @tool
# def update_plan(
#     update_todos: list[Todo],
#     tool_call_id: Annotated[str, InjectedToolCallId],
#     state: Annotated[State, InjectedState],
# ):
#     """用于更新todo任务状态的工具，可以多次使用来更新任务进度。

#     参数：
#     update_todos: list[Todo] - 需要更新的todo列表，每个元素是一个包含以下字段的字典：
#         - content: str, todo任务内容，必须与现有任务内容完全一致
#         - status: str, 任务状态，只能是"in_progress"（进行中）或"done"（已完成）

#     使用说明：
#     1. 每次调用只需传入需要更新状态的任务，无需传入所有任务
#     2. 必须同时包含至少一个"done"状态的任务和至少一个"in_progress"状态的任务
#        - 将已完成的任务设置为"done"
#        - 将接下来要执行的任务设置为"in_progress"
#     3. content字段必须与现有任务内容精确匹配

#     示例：
#     假设当前任务列表为：
#     1. 待办1 (in_progress)
#     2. 待办2 (pending)
#     3. 待办3 (pending)

#     当完成"待办1"并准备开始"待办2"时，应传入：
#     [
#         {"content": "待办1", "status": "done"},
#         {"content": "待办2", "status": "in_progress"}
#     ]
#     """

#     todo_list = state["todo"] if "todo" in state else []

#     updated_todo_list = []

#     # 遍历update_todo,每遍历一个todo，就从todo_list中查找对应的todo，并更新其状态
#     for update_todo in update_todos:
#         for todo in todo_list:
#             if todo["content"] == update_todo["content"]:
#                 todo["status"] = update_todo["status"]
#                 updated_todo_list.append(todo)

#     # 检查是否所有的todo were updated
#     if len(updated_todo_list) < len(update_todos):
#         raise ValueError(
#             "未找到如下的todo:"
#             + ",".join(
#                 [
#                     todo["content"]
#                     for todo in update_todos
#                     if todo not in updated_todo_list
#                 ]
#             )
#             + "请检查todo列表是否正确，目前的todo列表为:"
#             + "\n".join(
#                 [todo["content"] for todo in todo_list if todo["status"] != "done"]
#             )
#         )

#     return Command(
#         update={
#             "todo": todo_list,
#             "messages": [
#                 ToolMessage(content="Todo list 更新成功", tool_call_id=tool_call_id)
#             ],
#         }
#     )


@tool
async def transfor_task_to_subagent(
    content: Annotated[
        str,
        "当前待执行的todo任务内容，必须与todo列表中待办事项的content字段完全一致，但是当子智能体执行的任务有误时，重试的时候可以适当改写",
    ],
):
    """用于执行todo任务的工具。

    参数：
    content: str, 待执行的todo任务内容，必须与todo列表中待办事项的content字段完全一致，但是当子智能体执行的任务有误时，重试的时候可以适当改写

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
    state: Annotated[State, InjectedState],
):
    """用于写入笔记的工具。

    参数：
    content: str, 笔记内容

    """
    if file_name in state["note"] if "note" in state else {}:
        notes = state["note"] if "note" in state else {}
        file_name = file_name + "_" + str(len(notes[file_name]))

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


# @tool
# def ls(state: Annotated[State, InjectedState]):
#     """列出所有已保存的笔记名称。

#     返回：
#     list[str]: 包含所有笔记文件名的列表

#     """
#     notes = state["note"] if "note" in state else {}
#     return list(notes.keys())


# @tool
# def query_note(file_name: str, state: Annotated[State, InjectedState]):
#     """查询笔记。

#     参数：
#     file_name:笔记名称

#     返回：
#     str, 查询的笔记内容

#     """
#     notes = state["note"] if "note" in state else {}

#     return notes.get(file_name, "未找到笔记名称")


@tool
def get_weather(city: str):
    """查询天气。

    参数：
    city:城市名称

    返回：
    str, 天气信息

    """
    return f"{city}的天气是晴天，温度是25度。"


async def tavily_search(query: Annotated[str, "要搜索的内容"]):
    """互联网搜索工具，用于获取最新的网络信息和资料。注意：为控制上下文长度和降低调用成本，每个任务执行过程中仅可调用一次此工具。"""
    tavily_search = TavilySearch(
        max_results=5,
    )
    result = await tavily_search.ainvoke({"query": query})
    return result
