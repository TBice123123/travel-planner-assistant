from dotenv import load_dotenv
from langgraph.graph.state import StateGraph

from src.agent.node import call_model, tool_node
from src.agent.state import State, StateInput
from src.agent.sub_agent.graph import build_sub_agent
from src.agent.utils.context import Context


load_dotenv(dotenv_path=".env", override=True)


def build_graph_with_langgraph_studio():
    graph = StateGraph(State, input_schema=StateInput, context_schema=Context)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("subagent", build_sub_agent())

    graph.add_edge("__start__", "call_model")
    graph.add_edge("tools", "call_model")
    graph.add_edge("subagent", "call_model")

    return graph.compile()
