from dotenv import load_dotenv
from langchain_dev_utils import register_model_provider
from langchain_qwq import ChatQwen
from langchain_siliconflow import ChatSiliconFlow

load_dotenv(dotenv_path=".env", override=True)


from src.agent.graph import build_graph_with_langgraph_studio  # noqa: E402

__all__ = ["build_graph_with_langgraph_studio"]


register_model_provider("dashscope", ChatQwen)
register_model_provider("siliconflow", ChatSiliconFlow)
register_model_provider(
    "zai", "openai", base_url="https://open.bigmodel.cn/api/paas/v4"
)
register_model_provider("moonshot", "openai", base_url="https://api.moonshot.cn/v1")
