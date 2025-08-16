import argparse
import ast
import asyncio
import json
import traceback
from functools import wraps
from typing import Annotated, Any, Callable, Dict, List, Union
from dotenv import load_dotenv

from icecream import ic
from pydantic import BaseModel, Field, AfterValidator
from pydantic_ai import Agent

# MCP Imports
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from io import StringIO
import sys
import ast
import astunparse


# --- Asynchronous Python Executor (Unchanged) ---
async def aexec_python_local(
    code: str, globals: Dict = None, locals: Dict = None
) -> Dict[str, Any]:

    if globals is None:
        globals = {}
    if locals is None:
        locals = {}
    globals["asyncio"] = asyncio
    globals["__file__"] = __file__
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = redirected_stdout = StringIO()
    sys.stderr = redirected_stderr = StringIO()

    # Wrap a single expression in print() to see its value
    try:
        module = ast.parse(code)
        if len(module.body) == 1 and isinstance(module.body[0], ast.Expr):
            print_value = ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[module.body[0].value],
                keywords=[],
            )
            module.body[0] = ast.Expr(value=print_value)
        modified_code = astunparse.unparse(module)
        modified_code = f"{modified_code}\nglobals().update(locals())"
    except SyntaxError:
        # If parsing fails, it might be a multi-line statement.
        # Run it as-is and rely on the user to use print().
        modified_code = code

    wrapped_code = "async def __aexec_wrapper__():\n" + "".join(
        f"    {line}\n" for line in modified_code.splitlines()
    )
    exec_namespace: Dict[str, Any] = {}
    try:
        exec(wrapped_code, globals, exec_namespace)
        result = await exec_namespace["__aexec_wrapper__"]()
        # The automatic print now happens inside the exec'd code,
        # so we no longer need to print the result here.
    except Exception:
        tb = traceback.format_exc()
        redirected_stderr.write(tb)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return {
        "stdout": redirected_stdout.getvalue(),
        "stderr": redirected_stderr.getvalue(),
    }


# --- REFACTORED: MCPToolInjector as an Async Context Manager ---
class MCPToolInjector:
    """
    Connects to an MCP server, discovers its tools, and creates callable Python functions.
    This class is an async context manager to ensure proper resource handling.
    """

    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
        self._session: ClientSession | None = None
        self._tools: List[types.Tool] = []
        self._streams = None

    async def __aenter__(self):
        """Initializes the connection to the MCP server and fetches the tool list."""
        ic(f"Connecting to MCP server at {self.mcp_url}...")
        self._streams = streamablehttp_client(self.mcp_url)
        read_stream, write_stream, _ = await self._streams.__aenter__()

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()

        await self._session.initialize()
        list_tools_result = await self._session.list_tools()
        self._tools = list_tools_result.tools
        ic(f"Successfully connected and found {len(self._tools)} tools.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection to the MCP server."""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
        if self._streams:
            await self._streams.__aexit__(exc_type, exc_val, exc_tb)
        ic("MCP connection closed.")

        # In MCPToolInjector class

    def _create_callable_for_tool(self, tool: types.Tool) -> Callable:
        """Dynamically creates an async function to wrap an MCP tool call."""

        # FIX #1: The wrapper now accepts *args and **kwargs to be more robust.
        # This prevents the initial positional argument error.
        async def mcp_tool_wrapper(*args, **kwargs):
            # If positional arguments are passed, map them to keyword arguments
            # based on the tool's schema. This makes the agent more robust.
            if args:
                arg_names = list(tool.inputSchema.get("properties", {}).keys())
                for i, arg_val in enumerate(args):
                    if i < len(arg_names):
                        kwargs[arg_names[i]] = arg_val

            # ic(f"Calling MCP tool '{tool.name}' with arguments: {kwargs}")
            tool_call_result = await self._session.call_tool(
                tool.name, arguments=kwargs
            )

            if not tool_call_result.content:
                return None

            # FIX #2: The return value is now parsed as JSON. This is the critical fix.
            # This solves the "string indices must be integers" error.
            if tool_call_result.content[0].type == "text":
                raw_text = tool_call_result.content[0].text
                try:
                    # Attempt to deserialize the text as JSON.
                    return json.loads(raw_text)
                except json.JSONDecodeError:
                    # If it's not valid JSON, return the raw text.
                    # This handles simple string returns like "success".
                    return raw_text

            # Fallback for other content types (e.g., binary data)
            return tool_call_result.content

        # Create a user-friendly docstring for the LLM from the OpenAPI schema
        schema = tool.inputSchema
        arg_docs_list = []
        if "properties" in schema:
            for name, prop in schema.get("properties", {}).items():
                arg_docs_list.append(
                    f"        - {name} ({prop.get('type', 'any')}): {prop.get('description', '')}"
                )
        arg_docs = "\n".join(arg_docs_list)
        docstring = f"{tool.description}\n\n    Args:\n{arg_docs}"

        mcp_tool_wrapper.__name__ = tool.name
        mcp_tool_wrapper.__doc__ = docstring
        return mcp_tool_wrapper

    def get_tool_namespace(self) -> Dict[str, Callable]:
        """Returns a dictionary of {tool_name: callable_async_function}."""
        if not self._tools:
            return {}
        return {tool.name: self._create_callable_for_tool(tool) for tool in self._tools}

    def get_tools_prompt_string(self) -> str:
        """Generates a markdown string describing available tools for the system prompt."""
        if not self._tools:
            return ""
        prompt_str = "\n\n**AVAILABLE EXTERNAL TOOLS (MCP):**\n"
        prompt_str += "You MUST use `await` when calling these functions (e.g., `result = await get_weather(city='London')`).\n"
        prompt_str += (
            "The return value of an `await` call is automatically printed.\n\n"
        )
        for tool in self._tools:
            # Use the generated docstring for a richer prompt
            callable_tool = self._create_callable_for_tool(tool)
            prompt_str += f"- `async def {tool.name}(...)`:\n"
            doc = "\n".join(
                [f"  {line}" for line in callable_tool.__doc__.strip().split("\n")]
            )
            prompt_str += f"{doc}\n\n"
        return prompt_str


# --- Pydantic Model (Unchanged) ---
def check_and_return_code(code_string: str) -> str:
    try:
        ast.parse(code_string)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    return code_string


class Code(BaseModel):
    reasoning: str = Field(
        description="A place for you to think about what you are doing, and to try to catch any mistakes before you make them."
    )
    code: Annotated[str, AfterValidator(check_and_return_code)] = Field(
        description="A string containing valid Python code to be executed."
    )

class Critique(BaseModel):
    reasoning: str = Field(
        description="A place for you to think about what you are doing, and to try to catch any mistakes before you make them."
    )
    mistakes: List[str] = Field(
        description="A list of mistakes, logical errors, or forgotten steps found in the assistant's plan or reasoning. If no mistakes are found, return an empty list."
    )
    suggestions: List[str] = Field(
        description="A list of suggested improvements to the assistant's plan or reasoning. If no suggestions are provided, return an empty list. These are meant to be non-critical changes to the plan or reasoning that will help the assistant to complete the task."
    )
    is_final_answer_valid: bool = Field(
        description="Set to true if the last message from the assistant is a valid, complete final answer to the user's original request. Otherwise, set to false."
    )

# --- Async SequentialCodeAgent (Unchanged) ---
class SequentialCodeAgent:
    def __init__(
        self,
        llm: str,
        max_turns: int = 10,
        mcp_tools_namespace: Dict = None,
        mcp_tools_prompt: str = "",
    ):
        self.agent = Agent(
            model=llm,
            system_prompt=self._create_system_prompt(mcp_tools_prompt, pex=False),
            output_type=Union[str, Code],
        )

        self.final_answer_agent = Agent(
            model=llm,
            system_prompt=self._create_system_prompt(mcp_tools_prompt, pex=False),
            output_type=str,
        )

        critic_system_prompt = (
            "You are a meticulous and ruthless critic. Your task is to review a conversation between a user and an AI assistant. "
            "Your sole purpose is to identify flaws in the AI's reasoning and plan. "
            "1. Carefully read the user's original request (the first message).\n"
            "2. Read the entire conversation history.\n"
            "3. Identify any steps the AI assistant has forgotten, any logical errors it has made, or any inefficient plans (e.g., not using parallel execution when possible).\n"
            "4. Check if the assistant's last message is a final answer. If it is, verify if it FULLY addresses the user's original request. Has any part of the request been ignored?\n"
            "5. Return your findings as a list of strings. Include suggestions for how to improve the plan or answer."
            "6. The agent is in an unsandboxed environment, so it is able to access any normal python globals (like `__file__`, `__name__`, libraries, functions, classes, etc."
        )

        self.critic_agent = Agent(
            model=llm, # You could use a more powerful model like gpt-4o for the critic if desired
            system_prompt=critic_system_prompt,
            output_type=Critique,
            result_retries=3,
        )

        self.namespace_globals = mcp_tools_namespace or {}
        self.namespace_globals["__file__"] = __file__
        self.namespace_locals = {}
        self.max_turns = max_turns
        self.history = []

    def _create_system_prompt(self, mcp_tools_prompt: str, pex=True) -> str:
        # *** MODIFICATION START ***
        base_prompt = (
            "You are a helpful AI assistant that writes and executes Python code to answer user questions.\n\n"
            "**INSTRUCTIONS:**\n"
            '1. You can write and execute Python code. To do so, output a JSON object like: `{"code": "print(\'hello\')"}`.\n'
            "2. The Python environment is STATELESS. Variables do not persist.\n"
            "3. Use `print()` to see the result of any operation. The return value of a single `await` call is automatically printed.\n"
            "4. When you have the final answer, respond with a plain string."
            "5. Due to design constraints, you have unsandboxed access to the python environment, so you can import modules, use asyncio, "
            "read files, subprocesses, etc. If you wanted to, you could probably take over the computer (**don't, but you probably could**)\n"
            "6. Finally, I have some very important instructions for you: **DON'T BE STUPID.**\n"
        )

        # New, explicit instructions for parallel execution.
        parallel_execution_prompt = (
            "\n\n**IMPORTANT - PARALLEL EXECUTION:**\n"
            "To perform multiple tool calls in parallel for efficiency (e.g., sending multiple emails at once), "
            "you MUST use `asyncio.gather`. Do NOT `await` calls one by one if they can be run in parallel.\n\n"
            "Example of correct parallel execution:\n"
            "```python\n"
            "await asyncio.gather(\n"
            "    send_email(to='user1@example.com', subject='Subject 1', body='Body 1'),\n"
            "    send_email(to='user2@example.com', subject='Subject 2', body='Body 2')\n"
            ")\n"
            "```"
        )
        # Combine the prompts. The specific tool docs go first, then the general parallel instruction.
        return (
            base_prompt + mcp_tools_prompt + (parallel_execution_prompt if pex else "")
        )

    async def _execute_code_in_sandbox(self, code: str) -> Dict[str, Any]:
        ic("Passing code to aexec_python_local with stateful namespace")
        return await aexec_python_local(
            code=code,
            globals=self.namespace_globals,
            locals=self.namespace_locals,
        )

    async def run(self, prompt: str):
        # The user's initial prompt is the starting point.
        user_input = f"Here is your task:\n\n---\n{prompt}\n---"
        self.history = []  # Reset history for each new run

        for i in range(self.max_turns):
            ic(f"--- Turn {i + 1}/{self.max_turns} ---")

            # --- Main Agent's Turn ---
            # The agent runs with the latest user_input and the existing history.
            response = await self.agent.run(
                user_prompt=user_input, message_history=self.history
            )
            # We always update the history with the full conversation from the latest run.
            self.history = response.all_messages()

            # --- CRITIC'S TURN ---
            ic("--- Critic Reviewing Plan ---")
            critique_response = await self.critic_agent.run(
                message_history=response.all_messages()
            )
            critique = critique_response.output
            ic(critique)

            # --- DECISION POINT: Correct or Execute? ---

            # Case 1: The critic found mistakes.
            if critique.mistakes:
                ic("Critic found mistakes! Sending back for correction.")
                # Formulate a new, corrective prompt that will be the 'user_input' for the next loop.
                mistakes = '\n- '.join(critique.mistakes)
                user_input = (
                    "A supervising critic has reviewed your plan and found the following flaws. "
                    "You MUST address these issues in your next step. DO NOT ignore them.\n\n"
                    f"**CRITIC'S FEEDBACK:**\n- {mistakes}\n\n"
                    "Please provide a new, corrected plan and the corresponding code."
                )
                # Continue to the next turn to let the agent process the critique.
                continue

            # Case 2: The critic found no mistakes. Proceed with the agent's proposed action.
            ic("Critic found no mistakes. Proceeding with proposed action.")

            if isinstance(response.output, Code):
                ic(response.output.reasoning)
                code_to_run = response.output.code
                ic(f"Executing code:\n---\n{code_to_run}\n---")

                # Handle empty code block
                if not code_to_run.strip():
                    ic("Agent proposed no code to run. Asking for final answer.")
                    final_answer = await self.final_answer_agent.run(
                        user_prompt="What is the final answer?",
                        message_history=self.history,
                    )
                    return final_answer.output

                # Execute the code
                python_result = await self._execute_code_in_sandbox(code=code_to_run)

                # Format the execution results to be the 'user_input' for the next turn.
                stdout = python_result.get("stdout", "").strip()
                stderr = python_result.get("stderr", "").strip()

                if not stderr:
                    ic(python_result)
                else:
                    ic("Execution resulted in an error:")
                    ic(stderr)

                user_input = "The code execution produced the following output. Continue with your task."
                if stdout:
                    user_input += f"\n\nSTDOUT:\n```\n{stdout}\n```"
                if stderr:
                    user_input += f"\n\nSTDERR:\n```\n{stderr}\n```"
                if not stdout and not stderr:
                    user_input += "\n\nNOTE: The code ran without error and produced no output."

            elif isinstance(response.output, str):
                # Case 3: The agent provided a final answer.
                # The critic has already validated it (since critique.mistakes was empty).
                if critique.is_final_answer_valid:
                    ic(f"Final answer received and validated: {response.output}")
                    return response.output
                else:
                    # This case handles when the agent gives up prematurely.
                    ic("Critic invalidated the final answer. Sending back for more work.")
                    user_input = (
                        "A supervising critic has reviewed your final answer and determined it is INCOMPLETE. "
                        "You have not finished all parts of the original request. Please continue working."
                    )
                    continue

            else:
                # Error case
                error_message = f"Error: Unexpected response type from agent: {type(response.output)}"
                ic(error_message)
                return error_message

        print("Agent reached maximum turns without providing a final answer.")
        return None

# --- REFACTORED: Main usage with 'async with' ---
parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str)
async def main():
    mcp_url = "http://localhost:8001/mcp"
    args = parser.parse_args()
    try:
        async with (MCPToolInjector(mcp_url=mcp_url) as injector):
            tool_namespace = injector.get_tool_namespace()
            tool_prompt = injector.get_tools_prompt_string()

            agent = SequentialCodeAgent(
                llm="openai:gpt-4o-mini",
                mcp_tools_namespace=tool_namespace,
                mcp_tools_prompt=tool_prompt,
            )

            # Using a more explicit prompt to reduce ambiguity for the agent.
            # prompt = (
            # "Find me 2 new potential clients for commercial liability insurance in Austin, TX."
            # "They should be in the food and beverage industry. Get their owner's contact info, "
            # "add them to the CRM, and draft a personalized introductory email for each."
            # )
            prompt = args.prompt or "The owner of 'Austin's Artisan Bakery' has approved the proposal we generated at ./proposals/austins_artisan_bakery_proposal.pdf. Please send this document to austin.artisan@example.com for an e-signature. After sending it, schedule a 15-minute follow-up call with them for a 'Policy Onboarding Walkthrough'."
            ic(prompt)
            final_result = await agent.run(prompt)
            print("\n--- AGENT'S FINAL RESPONSE ---")
            print(final_result)

    except ConnectionError as e:
        print(f"\n[ERROR] Could not run agent: {e}")
        print(
            "Please ensure the MCP server (e.g., the reference 'examples/python/tools_server.py') is running."
        )
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()  # Print full traceback for easier debugging.
        print("Please ensure your OPENAI_API_KEY is set as an environment variable.")


if __name__ == "__main__":
    load_dotenv(".env", override=True)
    asyncio.run(main())