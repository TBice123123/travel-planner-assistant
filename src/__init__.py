from langchain_dev_utils import register_model_provider
from langchain_qwq import ChatQwen
from langchain_siliconflow import ChatSiliconFlow


register_model_provider("dashscope", ChatQwen)
register_model_provider("siliconflow", ChatSiliconFlow)
register_model_provider(
    "zai", "openai", base_url="https://open.bigmodel.cn/api/paas/v4"
)
register_model_provider("moonshot", "openai", base_url="https://api.moonshot.cn/v1")
register_model_provider(
    "deepseek",
    "deepseek",
    base_url="https://api.deepseek.com/v1",
)
