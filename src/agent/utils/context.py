from dataclasses import field, dataclass

from src.agent.prompts.prompt import (
    SUBAGENT_PROMPT,
    SUMMARY_PROMPT,
    TODO_MODEL_PROMPT,
    WRITE_PROMPT,
)


@dataclass
class Context:
    todo_model: str = field(
        default="dashscope:qwen3-235b-a22b-instruct-2507",
        metadata={
            "description": "用于执行todo任务规划和执行的模型",
            "name": "todo_model",
            "provider": "dashscope",
        },
    )
    sub_model: str = field(
        default="zai:glm-4.5",
        metadata={
            "description": "用于执行每个任务的模型",
            "name": "sub_model",
            "provider": "zai",
        },
    )
    note_model: str = field(
        default="dashscope:qwen-flash",
        metadata={
            "description": "用于执行记笔记任务的模型",
            "name": "note_model",
            "provider": "dashscope",
        },
    )
    summary_model: str = field(
        default="dashscope:qwen-flash",
        metadata={
            "description": "用于执行总结任务的模型",
            "name": "summary_model",
            "provider": "dashscope",
        },
    )
    todo_prompt: str = field(
        default=TODO_MODEL_PROMPT,
        metadata={
            "description": "用于执行todo任务规划和执行的模型",
            "name": "todo_prompt",
        },
    )
    sub_prompt: str = field(
        default=SUBAGENT_PROMPT,
        metadata={
            "description": "用于执行每个任务的模型",
            "name": "sub_prompt",
        },
    )
    note_prompt: str = field(
        default=WRITE_PROMPT,
        metadata={
            "description": "用于执行记笔记任务的模型",
            "name": "note_prompt",
        },
    )
    summary_prompt: str = field(
        default=SUMMARY_PROMPT,
        metadata={
            "description": "用于执行总结任务的模型",
            "name": "summary_prompt",
        },
    )
