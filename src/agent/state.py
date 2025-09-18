from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import MessagesState, add_messages
from langchain_dev_utils import PlanStateMixin, NoteStateMixin


def file_reducer(l: dict | None, r: dict | None):  # noqa: E741
    if l is None:
        return r
    elif r is None:
        return l
    else:
        return {**l, **r}


class StateInput(MessagesState):
    pass


class State(MessagesState, PlanStateMixin, NoteStateMixin, total=False):
    task_messages: Annotated[list[AnyMessage], add_messages]
    now_task_message_index: int
    write_note_messages: Annotated[list[AnyMessage], add_messages]
