from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)
from langchain_core.output_parsers import JsonOutputParser
from icecream import ic
import re
import ast

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

class SequentialCodeAgent:
    def __init__(self, options: SequentialCodeAgentOptions, execute:function):
        self.options = options
        self.llm = self.options.llm.bind(stop="```\n")
        self.execute = execute

        self.msgs = [
            SystemMessage(
                f""
            )
        ]

    
    def run(self):
        c = self.llm.invoke("Generate a simple python script.").content
        ic(c)
        
        # Use extract_valid_python_blocks to extract all valid code blocks
        valid_blocks = extract_valid_python_blocks(c)
        
        if not valid_blocks:
            raise ValueError("No valid code blocks found")

        # Take the last valid code block
        code_block = valid_blocks[-1][0].strip()
        ic(code_block)

        result = self.execute(code_block)

        ic(result)


        return code_block