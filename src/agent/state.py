from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import MessagesState, add_messages
from langchain_dev_utils import PlanStateMixin, NoteStateMixin


class StateInput(MessagesState):
    pass


class State(MessagesState, PlanStateMixin, NoteStateMixin, total=False):
    task_messages: Annotated[list[AnyMessage], add_messages]
