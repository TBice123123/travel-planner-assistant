from langchain_dev_utils import batch_register_model_provider
from langchain_qwq import ChatQwen
from langchain_siliconflow import ChatSiliconFlow


batch_register_model_provider(
    [
        {
            "provider": "dashscope",
            "chat_model": ChatQwen,
        },
        {
            "provider": "siliconflow",
            "chat_model": ChatSiliconFlow,
        },
        {
            "provider": "zai",
            "chat_model": "openai",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
        },
        {
            "provider": "moonshot",
            "chat_model": "openai",
            "base_url": "https://api.moonshot.cn/v1",
        },
    ]
)
