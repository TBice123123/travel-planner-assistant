from dataclasses import dataclass
from typing import Annotated

from src.agent.prompts.prompt import (
    SUBAGENT_PROMPT,
    SUMMARY_PROMPT,
    TODO_MODEL_PROMPT,
    WRITE_PROMPT,
)


@dataclass
class Context:
    todo_model: Annotated[str, "用于执行todo任务规划和执行的模型"] = (
        "moonshot:kimi-k2-0905-preview"
    )
    sub_model: Annotated[str, "用于执行每个任务的模型"] = "deepseek:deepseek-chat"
    write_model: Annotated[str, "用于执行记笔记任务的模型"] = (
        "dashscope:qwen3-next-80b-a3b-instruct"
    )
    summary_model: Annotated[str, "用于执行总结任务的模型"] = "zai:glm-4.5-air"
    todo_prompt: Annotated[str, "用于执行todo任务的prompt"] = TODO_MODEL_PROMPT
    sub_prompt: Annotated[str, "用于执行每个任务的prompt"] = SUBAGENT_PROMPT
    write_prompt: Annotated[str, "用于执行记笔记任务的prompt"] = WRITE_PROMPT
    summary_prompt: Annotated[str, "用于执行总结任务的prompt"] = SUMMARY_PROMPT
