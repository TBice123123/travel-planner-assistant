from langgraph.graph.state import StateGraph

from src.agent.sub_agent.state import SubAgentState
from src.agent.sub_agent.node import sub_tools, subagent_call_model
from src.agent.sub_agent.write_summary.node import build_write_and_summary_node


def build_sub_agent():
    subgraph = StateGraph(SubAgentState)
    subgraph.add_node("subagent_call_model", subagent_call_model)
    subgraph.add_node("sub_tools", sub_tools)
    subgraph.add_node("write_and_summary", build_write_and_summary_node())
    subgraph.add_edge("__start__", "subagent_call_model")
    subgraph.add_edge("sub_tools", "subagent_call_model")

    subgraph.add_edge("write_and_summary", "subagent_call_model")

    return subgraph.compile()
