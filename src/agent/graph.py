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
    # https://docs.langchain.com/oss/python/langgraph/use-subgraphs#full-example-shared-state-schemas
    # 子图状态于父亲图转台不完全一致
    # LangGraph 的处理方式：
    # ✅ 主Agent的所有字段（messages, task_messages, plan, note 等）会完整传递给子Agent
    # ✅ 子Agent特有的字段（temp_task_messages）会被自动初始化为默认值（空列表）
    # ✅ 这就像 Python 的子类实例化：父类字段有值，子类新增字段用默认值
    graph.add_node("subagent", build_sub_agent())  # 子agent节点

    graph.add_edge("__start__", "call_model")
    graph.add_edge("tools", "call_model")
    graph.add_edge("subagent", "call_model")

    return graph.compile()
