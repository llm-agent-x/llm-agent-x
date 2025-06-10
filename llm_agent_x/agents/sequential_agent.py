import json
import traceback
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from icecream import ic
from typing import Callable, Any, Dict, List, Type

from llm_agent_x.tools.exec_python import exec_python

class CodeExecutor:
    """Safely executes Python code using the sandboxed exec_python tool."""
    def __init__(self, tools: List[Callable] = None):
        self.tool_schema = convert_to_openai_tool(exec_python)

    def execute(self, code: str) -> Dict[str, Any]:
        """Executes the given code via the sandbox and returns the resulting dictionary."""
        ic("Passing code to sandboxed exec_python")
        return exec_python(code=code)

class SequentialCodeAgent:
    def __init__(self, llm: BaseChatModel, tools: List[Callable] = None):
        self.llm = llm
        self.tools = tools or []
        self.executor = CodeExecutor()
        self.llm_with_tools = self.llm.bind_tools([self.executor.tool_schema])
        self.system_prompt = self._create_system_prompt()
        self.msgs = [SystemMessage(content=self.system_prompt)]

    # --- CHANGE #1: A new system prompt that correctly describes the STATELESS sandbox ---
    def _create_system_prompt(self) -> str:
        """Generates the system prompt with correct instructions for the stateless sandbox."""
        base_prompt = (
            "You are a helpful AI assistant that writes and executes Python code in a secure, sandboxed environment to answer questions.\n\n"
            "**CRITICAL INSTRUCTIONS FOR USING THE SANDBOX:**\n"
            "1. You have one tool: `exec_python`.\n"
            "2. The sandbox is **STATELESS**. Each call to `exec_python` is a completely new environment. Variables DO NOT persist between calls.\n"
            "3. To perform multi-step tasks, you **MUST** carry the context from previous steps into the next code block. For example, if you retrieve a value in step 1, you must re-declare it as a variable in step 2.\n"
            "4. To see the result of any operation, you **MUST** `print()` it. The content of `stdout` will be returned to you.\n"
            "5. The sandbox does not have access to any functions unless you define them within the code you provide.\n\n"
            "When you have the final answer, respond directly to the user."
        )

        if not self.tools:
            return base_prompt

        tool_descriptions = "\n\n**HELPER FUNCTION DEFINITIONS:**\n"
        tool_descriptions += "To use the following helper functions, you must copy their complete function definition into the code block you send to `exec_python`.\n"
        for tool in self.tools:
            import inspect
            try:
                source_code = inspect.getsource(tool)
                tool_descriptions += f"\n```python\n{source_code}```\n"
            except (TypeError, OSError):
                tool_descriptions += f"# Could not retrieve source for {tool.__name__}\n"

        return base_prompt + tool_descriptions

    # --- CHANGE #2: A more robust formatter that cleans up output ---
    def _format_sandbox_result(self, result: Dict[str, Any]) -> str:
        """Formats the dictionary from the sandbox into a simple string for the LLM."""
        if not isinstance(result, dict):
            ic("Warning: Sandbox result was not a dictionary.", result)
            return f"An unexpected error occurred. Tool returned: {str(result)}"

        # Clean up stdout by removing empty lines and lines that are exactly 'None'
        stdout_raw = result.get('stdout', '')
        stdout_lines = [line for line in stdout_raw.strip().split('\n') if line.strip() and line.strip() != 'None']
        stdout = '\n'.join(stdout_lines)

        stderr = result.get('stderr', '').strip()

        if stderr:
            return f"Execution failed with an error.\nSTDERR:\n{stderr}"
        elif stdout:
            return f"Execution successful.\nSTDOUT:\n{stdout}"
        else:
            return "Execution successful. No output was produced on stdout."

    def reset(self):
        """Resets the conversation history."""
        ic("Resetting agent state.")
        self.msgs = [SystemMessage(content=self.system_prompt)]

    def run(self, prompt: str):
        """Runs the agent loop for a given prompt."""
        self.msgs.append(HumanMessage(content=prompt))
        max_turns = 10
        for _ in range(max_turns):
            ic(f"Invoking LLM with {len(self.msgs)} messages...")
            response: AIMessage = self.llm_with_tools.invoke(self.msgs)
            self.msgs.append(response)

            if not response.tool_calls:
                ic("LLM provided a final answer.")
                return response.content

            ic(f"LLM requested {len(response.tool_calls)} tool calls.")
            for tool_call in response.tool_calls:
                if tool_call["name"] == self.executor.tool_schema['function']['name']:
                    code_to_execute = tool_call["args"]["code"]
                    ic("Code to be executed in sandbox:", code_to_execute)
                    sandbox_result = self.executor.execute(code_to_execute)
                    ic("Raw sandbox result:", sandbox_result)
                    formatted_result = self._format_sandbox_result(sandbox_result)
                    ic("Formatted result for LLM:", formatted_result)
                    self.msgs.append(ToolMessage(content=formatted_result, tool_call_id=tool_call["id"]))
                else:
                    error_msg = f"Error: Agent tried to call an unknown tool '{tool_call['name']}'. Only 'exec_python' is available."
                    self.msgs.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
        
        return "Agent stopped after reaching the maximum number of turns."

# --- Example Usage (no changes needed here) ---
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()

    def get_user_email_address(user_name: str) -> str:
        """Looks up the email address for a given user name."""
        email_db = {"Alice": "alice@example.com", "Bob": "bob@work.net"}
        return email_db.get(user_name, "User not found.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = SequentialCodeAgent(llm=llm, tools=[get_user_email_address])
    
    user_prompt = "What is the email for a user named Alice? Then, can you tell me the domain name of that email address?"
    ic.enable()
    final_answer = agent.run(user_prompt)
    print("\n--- FINAL AGENT ANSWER ---")
    print(final_answer)
    print("--------------------------")

    print("\n--- DEMONSTRATING ERROR HANDLING ---")
    agent.reset()
    user_prompt_fail = "Please divide 100 by 0 and tell me the result."
    final_answer_fail = agent.run(user_prompt_fail)
    print("\n--- FINAL AGENT ANSWER (FAILURE) ---")
    print(final_answer_fail)
    print("------------------------------------")