from langgraph.graph.state import StateGraph
from src.agent.state import State
from src.agent.write_agent.node import write, write_tool, summary


def build_write_agent():
    subgraph = StateGraph(State)
    subgraph.add_node("write", write)
    subgraph.add_node("write_tool", write_tool)
    subgraph.add_node("summary", summary)
    subgraph.add_edge("__start__", "write")
    subgraph.add_edge("__start__", "summary")
    subgraph.add_edge("write", "write_tool")
    subgraph.add_edge("summary", "write_tool")
    subgraph.add_edge("write_tool", "__end__")

    return subgraph.compile()
