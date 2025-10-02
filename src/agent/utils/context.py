from dataclasses import dataclass
from typing import Annotated

from src.agent.prompts.prompt import (
    SUBAGENT_PROMPT,
    SUMMARY_PROMPT,
    PLAN_MODEL_PROMPT,
)


@dataclass
class Context:
    plan_model: Annotated[str, "用于执行任务规划的模型"] = (
        "moonshot:kimi-k2-0905-preview"
    )
    sub_model: Annotated[str, "用于执行每个任务的模型"] = "deepseek:deepseek-chat"
    summary_model: Annotated[str, "用于执行总结任务的模型"] = "dashscope:qwen-flash"
    plan_prompt: Annotated[str, "用于执行任务规划的prompt"] = PLAN_MODEL_PROMPT
    sub_prompt: Annotated[str, "用于执行每个任务的prompt"] = SUBAGENT_PROMPT
    summary_prompt: Annotated[str, "用于执行总结任务的prompt"] = SUMMARY_PROMPT
