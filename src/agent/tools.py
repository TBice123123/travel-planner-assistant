from typing import Annotated

from langchain_core.tools import tool
from langchain_tavily.tavily_search import TavilySearch

from langchain_dev_utils import (
    create_update_note_tool,
    create_write_note_tool,
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

write_note = create_write_note_tool(
    name="write_note",
    description="""用于写入笔记的工具。

    参数：
    content: str, 笔记内容

    """,
    message_key="write_note_messages",
)

update_note = create_update_note_tool(
    name="update_note",
    description="""用于更新笔记的工具。

    参数：
    file_name: str, 笔记名称
    orignal_content: str, 笔记原始内容
    new_content: str, 笔记更新后的内容
    """,
    message_key="write_note_messages",
)


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
