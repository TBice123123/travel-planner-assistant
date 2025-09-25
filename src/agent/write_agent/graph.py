from langgraph.graph.state import StateGraph

from src.agent.write_agent.state import WriteState
from src.agent.write_agent.node import summary, write, write_tool


def build_write_agent():
    subgraph = StateGraph(WriteState)
    subgraph.add_node("write", write)
    subgraph.add_node("write_tool", write_tool)
    subgraph.add_node("summary", summary)
    subgraph.add_edge("__start__", "write")
    subgraph.add_edge("__start__", "summary")
    subgraph.add_edge("write", "write_tool")
    subgraph.add_edge("summary", "__end__")
    subgraph.add_edge("write_tool", "__end__")

    return subgraph.compile()
