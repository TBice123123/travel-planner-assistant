from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from src.agent.state import State


class WriteState(State):
    temp_write_note_messages: Annotated[list[AnyMessage], add_messages]
