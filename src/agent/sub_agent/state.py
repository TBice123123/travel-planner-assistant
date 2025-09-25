from typing import Annotated
from langchain_core.messages import AnyMessage
from src.agent.state import State
from langgraph.graph.message import add_messages


class SubAgentState(State):
    temp_task_messages: Annotated[list[AnyMessage], add_messages]
