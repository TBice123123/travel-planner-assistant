import os
from typing import Any, Literal
from langchain.chat_models import init_chat_model
from langchain_qwq import ChatQwen


def _load_qwen_model(model_name: str, **kwargs: Any):
    return ChatQwen(model=model_name, **kwargs)


def load_chat_model(
    model_name: str,
    model_provider: Literal["deepseek", "dashscope", "siliconflow"],
    **kwargs: Any,
):
    if model_provider == "deepseek":
        return init_chat_model(model_name, model_provider=model_provider, **kwargs)
    elif model_provider == "dashscope":
        return _load_qwen_model(model_name, **kwargs)
    elif model_provider == "siliconflow":
        if api_key := os.getenv("SILICONFLOW_API_KEY"):
            kwargs["api_key"] = api_key
        if "base_url" not in kwargs:
            kwargs["base_url"] = (
                os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
            )
        return init_chat_model(model_name, model_provider="openai", **kwargs)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
