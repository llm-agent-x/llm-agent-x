import ast
import traceback
from pydantic import BaseModel, Field, AfterValidator
from pydantic_ai import Agent
from icecream import ic
from typing import Annotated, Any, Dict, List, Union

# Mock implementation for self-contained example
def exec_python_local(code: str, globals: Dict = None, locals: Dict = None) -> Dict[str, Any]:
    from io import StringIO
    import sys
    if globals is None: globals = {}
    if locals is None: locals = {}
    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_stdout = sys.stdout = StringIO()
    redirected_stderr = sys.stderr = StringIO()
    try:
        exec(code, globals, locals)
    except Exception:
        tb = traceback.format_exc()
        redirected_stderr.write(tb)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return { "stdout": redirected_stdout.getvalue(), "stderr": redirected_stderr.getvalue() }

# --- THE FIX: A validator that checks syntax but returns the original string ---
def check_and_return_code(code_string: str) -> str:
    """
    Validates that the input string is syntactically correct Python.
    If it is, it returns the original string.
    If not, it raises a ValueError, which Pydantic handles.
    """
    try:
        ast.parse(code_string)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    return code_string

# --- CHANGE: Update the Pydantic model to use the new validator ---
class Code(BaseModel):
    """A Pydantic model for a string of Python code that is validated to be syntactically correct."""
    code: Annotated[str, AfterValidator(check_and_return_code)] = Field(
        description="A string containing valid Python code to be executed."
    )

class SequentialCodeAgent:
    def __init__(self, llm: str, max_turns: int = 5):
        self.agent = Agent(
            model=llm,
            system_prompt=self._create_system_prompt(),
            output_type=Union[str, Code]
        )
        self.namespace_globals = {}
        self.namespace_locals = {}
        self.max_turns = max_turns
        self.history = []

    def _create_system_prompt(self) -> str:
        """Generates the system prompt with correct instructions for the stateful sandbox."""
        return (
            "You are a helpful AI assistant that writes and executes Python code to answer user questions.\n\n"
            "**INSTRUCTIONS:**\n"
            "1. You can write and execute Python code to help you solve the problem.\n"
            "2. To execute code, you must output a JSON object with a single key 'code' containing the Python code as a string. For example: `{\"code\": \"x = 10\\nprint(x * 2)\"}`.\n"
            "3. The Python execution environment is STATEFUL. Any variables, functions, or imports you define will persist in subsequent code executions.\n"
            "4. The output of the code execution (`stdout` and `stderr`) will be returned to you in the next turn. You MUST use the `print()` function to see the result of any operation.\n"
            "5. When you have the final answer, or if you need to ask the user a clarifying question, respond with a plain string instead of a code JSON object."
        )

    def _execute_code_in_sandbox(self, code: str) -> Dict[str, Any]:
        """Executes code using the stateful namespaces."""
        ic("Passing code to exec_python_local with stateful namespace")
        return exec_python_local(code=code, globals=self.namespace_globals, locals=self.namespace_locals)

    def run(self, prompt: str):
        """Runs the agent loop for a given prompt."""
        current_prompt = f"Here is your task: \n\n\"{prompt}\""
        for i in range(self.max_turns):
            ic(f"--- Turn {i+1}/{self.max_turns} ---")
            response = self.agent.run_sync(current_prompt, message_history=self.history)
            self.history = response.all_messages()
            if isinstance(response.output, Code):
                # Now, response.output.code is guaranteed to be a string
                code_to_run = response.output.code
                ic(f"Executing code:\n---\n{code_to_run}\n---")
                python_result = self._execute_code_in_sandbox(code=code_to_run)
                ic(f"Execution result: {python_result}")
                stdout = python_result.get('stdout', '').strip()
                stderr = python_result.get('stderr', '').strip()
                current_prompt = "The code execution produced the following output. Continue with your task."
                if stdout:
                    current_prompt += f"\n\nSTDOUT:\n```\n{stdout}\n```"
                if stderr:
                    current_prompt += f"\n\nSTDERR:\n```\n{stderr}\n```"
                if not stdout and not stderr:
                    current_prompt += "\n\nNOTE: The code ran without error and produced no output to stdout."
            elif isinstance(response.output, str):
                final_answer = response.output
                ic(f"Final answer received: {final_answer}")
                return final_answer
            else:
                error_message = f"Error: Unexpected response type from agent: {type(response.output)}"
                ic(error_message)
                return error_message
        print("Agent reached maximum turns without providing a final answer.")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    try:
        agent = SequentialCodeAgent(llm="openai:gpt-4o-mini")
        final_result = agent.run(
            "First, create a list of the first 5 prime numbers and assign it to a variable called `primes`. "
            "Then, in a second step, print each of those prime numbers multiplied by 2."
        )
        print("\n--- AGENT'S FINAL RESPONSE ---")
        print(final_result)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable.")