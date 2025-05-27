import json
import uuid
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)
from icecream import ic
import re
import ast
from typing import Callable, Any, Dict


def is_valid_python(code):
    """
    Check if the provided string is valid Python code using the AST module.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_valid_python_blocks(text):
    """
    Extract all valid Python code blocks and check if they have matching triple backticks.
    Returns a list of tuples containing the code block and whether it has matching backticks.
    """
    matches = list(re.finditer(r"```python", text))
    valid_code_blocks = []

    for match in reversed(matches):
        start_index = match.end()  # Start right after ` ```python `
        code = text[start_index:]  # Slice text from this point onward

        # Check for closing triple backticks
        end_index = code.find("```")
        if end_index != -1:
            code_block = code[:end_index].strip()
            has_matching_backticks = True
        else:
            code_block = code.strip()
            has_matching_backticks = False

        # Validate the code block
        if is_valid_python(code_block):
            valid_code_blocks.append((code_block, has_matching_backticks))

    return valid_code_blocks[::-1]  # Return results in original order


class SequentialCodeAgentOptions(BaseModel):
    llm: BaseChatModel = None


class IsDone(BaseModel):
    is_complete: bool = Field(
        description="Whether or not the conversation history has enough context to answer the original user question, or if the original user task is complete"
    )


class SequentialCodeAgent:
    def __init__(
        self, options: SequentialCodeAgentOptions, execute: Callable[[str, Dict], Any]
    ):
        self.options = options
        self.llm = self.options.llm.bind(stop="```\n")

        self.continue_llm = self.options.llm.with_structured_output(schema=IsDone)
        self.execute = execute

        self.msgs = [
            SystemMessage(
                f"You are an AI agent. Your task is to assist the user. To help you do that, you have the ability "
                f"to execute python. If you include a valid python code block, it will be executed with `exec()`. "
                f"Keep in mind, you can't access a filesystem. You can't make any system calls. "
                f"if the user provides you any context, it will be saved in a variable `context`. "
                f"When you need to use this variable, first include a block of code to understand the "
                f"structure of the object. There are also functions that you can call, to help complete tasks. "
                f"These will be specified by the user. Keep in mind, the user doesn't need to run any code. "
                f"As soon as you output the *closing triple backticks*, your generation will be paused, and a "
                f"user message will be appended to give you the results. Once you get the result to the python "
                f"code block, just answer the question for the user if you have enough to answer the question. "
                f"If an entry in the function list has type hints, don't use code to attempt to understand the structure. "
                f"If you need to see the result of a codeblock, assign it to the `result` variable, and the system will show you "
                f"the response after execution. "
            )
        ]
        self.code_execution_starting_namespace = {}
        self.code_execution_namespace = {}
        self.code_execution_namespace.update(self.code_execution_starting_namespace)

    def reset_code_execution_namespace(self):
        self.code_execution_namespace.clear()
        self.code_execution_namespace.update(self.code_execution_starting_namespace)

    def run(self, prompt):
        self.msgs.append(
            HumanMessage(
                f"USER: {prompt}\n\n"
                f"If it helps you, this is the state of the global variables in the code execution namespace: \n\n"
                f"{json.dumps(self.code_execution_namespace)}"
            )
        )

        done = False
        while not done:
            c = self.llm.invoke(self.msgs).content
            response = AIMessage(f"{c}\n```")
            ic(c)

            # Use extract_valid_python_blocks to extract all valid code blocks
            valid_blocks = extract_valid_python_blocks(c)

            if not valid_blocks:
                return c

            # Take the last valid code block
            code_block = valid_blocks[-1][0].strip()
            ic(code_block)

            self.execute(code_block, self.code_execution_namespace)
            result = self.code_execution_namespace.get("result", None)
            if result is None:
                result = ValueError(
                    "Error executing code; This may be because the user didn't approve code execution, "
                    "because the system detected malicious code or denied code execution priveliges for "
                    "other reasons, or a catch-all code execution callback. "
                    "This could also be because the code didn't assign anything to the `result` variable. "
                    "This may or may not be normal behaviour. It is up to you to decide."
                )
            ic(result)

            self.msgs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=f"run_python_snippet_{str(uuid.uuid4())}",
                )
            )

            # Check if the task is complete

            complete = self.continue_llm.invoke(
                self.msgs[1:]
                + [
                    HumanMessage(
                        "Now, is this question sufficiently answered, or the task sufficiently completed? "
                        "Respond following this JSON schema:\n\n"
                        f"{IsDone.model_json_schema()}"
                    )
                ]
            )

            done = complete.is_complete

        return self.llm.invoke(self.msgs).content
