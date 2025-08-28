import os
from typing import Any, Literal, Optional
from langchain.chat_models import init_chat_model
from langchain_qwq import ChatQwen
from typing import cast


# 支持的模型提供商类型定义
# deepseek: DeepSeek API (深度求索)
# dashscope: 阿里云 DashScope 平台 (包含通义千问等模型)
# siliconflow: SiliconFlow 平台 (硅基流动，提供多种开源模型服务)
# moonshot: Moonshot AI (月之暗面，提供 Kimi 等大模型)
type ModelProvider = Literal["deepseek", "dashscope", "siliconflow", "moonshot"]


def _get_model_name_and_provider(
    model_name: str,
) -> tuple[Optional[ModelProvider], str]:
    # 解析格式如 "provider:model_name" 或 "model_name"
    if ":" in model_name:
        model_provider, model_name = model_name.split(":")
        model_provider = cast(ModelProvider, model_provider)
    else:
        model_provider = None
    return model_provider, model_name


def load_chat_model(
    model_name: str,
    model_provider: Optional[ModelProvider] = None,
    **kwargs: Any,
):
    if model_provider is None:
        model_provider, model_name = _get_model_name_and_provider(model_name)
    if model_provider is None:
        if "qwen" in model_name:
            model_provider = "dashscope"
        elif "deepseek" in model_name:
            model_provider = "deepseek"
        elif "moonshot" in model_name or "kimi" in model_name:
            model_provider = "moonshot"
        else:
            raise ValueError(
                f"Unsupported model name: {model_name}, please specify model_provider."
            )

    if model_provider == "deepseek":
        return init_chat_model(model_name, model_provider=model_provider, **kwargs)
    elif model_provider == "dashscope":
        kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return ChatQwen(model=model_name, **kwargs)
    elif model_provider == "moonshot":
        if api_key := os.getenv("MOONSHOT_API_KEY"):
            kwargs["api_key"] = api_key
        if "base_url" not in kwargs:
            kwargs["base_url"] = (
                os.getenv("MOONSHOT_BASE_URL") or "https://api.moonshot.cn/v1"
            )
        return init_chat_model(model_name, model_provider="openai", **kwargs)
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


if __name__ == "__main__":
    model = load_chat_model("kimi-k2-0711-preview")
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    struct_model = model.with_structured_output(User, method="function_calling")
    print(struct_model.invoke("my name is xiaoming and I am 18 years old"))
