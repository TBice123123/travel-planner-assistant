from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import MessagesState, add_messages


class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "done"]


def file_reducer(l: dict | None, r: dict | None):  # noqa: E741
    if l is None:
        return r
    elif r is None:
        return l
    else:
        return {**l, **r}


class StateInput(MessagesState):
    pass


class State(MessagesState, total=False):
    todo: list[Todo]
    task_messages: Annotated[list[AnyMessage], add_messages]
    note: Annotated[dict[str, str], file_reducer]
    now_task_message_index: int
    write_note_messages: Annotated[list[AnyMessage], add_messages]
