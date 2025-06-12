import ast
import asyncio
import traceback
from functools import wraps
from typing import Annotated, Any, Callable, Dict, List, Union

from icecream import ic
from pydantic import BaseModel, Field, AfterValidator
from pydantic_ai import Agent

# MCP Imports
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

# --- NEW: Asynchronous Python Executor ---
async def aexec_python_local(code: str, globals: Dict = None, locals: Dict = None) -> Dict[str, Any]:
    """
    Asynchronously executes a string of Python code, allowing for `await`.
    It captures stdout and stderr.
    """
    from io import StringIO
    import sys

    if globals is None:
        globals = {}
    if locals is None:
        locals = {}

    # Provide asyncio in the execution context
    globals['asyncio'] = asyncio

    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_stdout = sys.stdout = StringIO()
    redirected_stderr = sys.stderr = StringIO()

    # Wrap the user's code in an async function to allow top-level await
    wrapped_code = "async def __aexec_wrapper__():\n"
    wrapped_code += "".join(f"    {line}\n" for line in code.splitlines())

    try:
        # Define the wrapper function in the given namespace
        exec(wrapped_code, globals, locals)
        # Await the execution of the wrapper function
        result = await locals['__aexec_wrapper__']()

        # If the code returns a value, print it so the LLM can see the result.
        if result is not None:
            print(result)

    except Exception:
        tb = traceback.format_exc()
        redirected_stderr.write(tb)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return {"stdout": redirected_stdout.getvalue(), "stderr": redirected_stderr.getvalue()}

# --- NEW: MCP Tool Injector Class ---
class MCPToolInjector:
    """Connects to an MCP server, discovers its tools, and creates callable Python functions for them."""
    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
        self._session: ClientSession | None = None
        self._tools: List[types.Tool] = []
        self._streams = None

    async def connect(self):
        """Initializes the connection to the MCP server and fetches the tool list."""
        try:
            ic(f"Connecting to MCP server at {self.mcp_url}...")
            self._streams = streamablehttp_client(self.mcp_url)
            ic("Connected to MCP server.")
            read_stream, write_stream, _ = await self._streams.__aenter__()
            ic("Streams initialized.")

            self._session = ClientSession(read_stream, write_stream)
            ic("Session created.")
            await self._session.__aenter__()
            ic("Session entered.")
            await self._session.initialize()
            ic("Session initialized.")
            self._tools = await self._session.list_tools()
            self._tools = self._tools.tools
            ic(self._tools)
            # ic(f"Successfully connected and found {len(self._tools)} tools.")
        except Exception as e:
            import traceback
            traceback.print_exc()

            ic(f"Failed to connect to MCP server: {e}")
            self._tools = []
            if self._streams:
                await self._session.__aexit__(None, None, None)
                await self._streams.__aexit__(None, None, None)
            raise ConnectionError(f"Could not connect to or initialize MCP server at {self.mcp_url}") from e

    async def close(self):
        """Closes the connection to the MCP server."""
        if self._streams:
            await self._session.__aexit__(None, None, None)
            await self._streams.__aexit__(None, None, None)
            ic("MCP connection closed.")

    def _create_callable_for_tool(self, tool: types.Tool) -> Callable:
        """Dynamically creates an async function to wrap an MCP tool call."""
        @wraps(self._create_callable_for_tool) # Basic wrap, will be overwritten
        async def mcp_tool_wrapper(**kwargs):
            ic(f"Calling MCP tool '{tool.name}' with arguments: {kwargs}")
            # The result object from call_tool has a 'result' attribute
            tool_call_result = await self._session.call_tool(tool.name, arguments=kwargs)
            return tool_call_result.result

        # Create a user-friendly docstring for the LLM
        schema = tool.inputSchema
        arg_docs = "\n".join(
            f"        - {prop['title']} ({prop['type']}): {prop.get('description', '')}"
            for prop in schema.get("properties", {}).values()
        )
        docstring = f"{tool.description}\n\n    Args:\n{arg_docs}"
        
        mcp_tool_wrapper.__name__ = tool.name
        mcp_tool_wrapper.__doc__ = docstring

        return mcp_tool_wrapper

    def get_tool_namespace(self) -> Dict[str, Callable]:
        """Returns a dictionary of {tool_name: callable_async_function}."""
        if not self._tools:
            return {}
        ic("-- MCP Tools --")
        ic(self._tools[0])
        return {tool.name: self._create_callable_for_tool(tool) for tool in self._tools}

    def get_tools_prompt_string(self) -> str:
        """Generates a markdown string describing available tools for the system prompt."""
        if not self._tools:
            return ""
        
        prompt_str = "\n\n**AVAILABLE EXTERNAL TOOLS (MCP):**\n"
        prompt_str += "You can use the following special functions, which are already imported. "
        prompt_str += "You MUST use `await` when calling them (e.g., `result = await echo_tool(message='hello')`).\n\n"

        for tool in self._tools:
            prompt_str += f"- `async def {tool.name}(...)`:\n"
            prompt_str += f"  - Description: {tool.description}\n"
            # for arg in tool.arguments:
            #      prompt_str += f"  - Argument: `{arg.name}` (type: {arg.type}, required: {arg.required})\n"
        return prompt_str

# --- Pydantic Model (Unchanged) ---
def check_and_return_code(code_string: str) -> str:
    try:
        ast.parse(code_string)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    return code_string

class Code(BaseModel):
    code: Annotated[str, AfterValidator(check_and_return_code)] = Field(
        description="A string containing valid Python code to be executed."
    )

# --- REFACTORED: Async SequentialCodeAgent ---
class SequentialCodeAgent:
    def __init__(self, llm: str, max_turns: int = 5, mcp_tools_namespace: Dict = None, mcp_tools_prompt: str = ""):
        self.agent = Agent(
            model=llm,
            system_prompt=self._create_system_prompt(mcp_tools_prompt),
            output_type=Union[str, Code]
        )
        self.namespace_globals = mcp_tools_namespace or {}
        self.namespace_locals = {}
        self.max_turns = max_turns
        self.history = []

    def _create_system_prompt(self, mcp_tools_prompt: str) -> str:
        base_prompt = (
            "You are a helpful AI assistant that writes and executes Python code to answer user questions.\n\n"
            "**INSTRUCTIONS:**\n"
            "1. You can write and execute Python code to help you solve the problem.\n"
            "2. To execute code, output a JSON object with a single key 'code' containing the Python code as a string. For example: `{\"code\": \"x = 10\\nprint(x * 2)\"}`.\n"
            "3. The Python execution environment is STATEFUL. Any variables, functions, or imports you define will persist.\n"
            "4. The output of the code execution (`stdout` and `stderr`) will be returned to you. You MUST use the `print()` function to see the result of any operation. Asynchronous functions will have their return value automatically printed.\n"
            "5. When you have the final answer, respond with a plain string instead of a code JSON object."
        )
        return base_prompt + mcp_tools_prompt

    async def _execute_code_in_sandbox(self, code: str) -> Dict[str, Any]:
        """Executes code using the stateful namespaces, now with async support."""
        ic("Passing code to aexec_python_local with stateful namespace")
        return await aexec_python_local(code=code, globals=self.namespace_globals, locals=self.namespace_locals)

    async def run(self, prompt: str):
        """Runs the agent loop for a given prompt, now asynchronously."""
        current_prompt = f"Here is your task: \n\n\"{prompt}\""
        for i in range(self.max_turns):
            ic(f"--- Turn {i+1}/{self.max_turns} ---")
            # Use run for async execution instead of run_sync
            response = await self.agent.run(current_prompt, message_history=self.history)
            self.history = response.all_messages()
            if isinstance(response.output, Code):
                code_to_run = response.output.code
                ic(f"Executing code:\n---\n{code_to_run}\n---")
                python_result = await self._execute_code_in_sandbox(code=code_to_run)
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

# --- UPDATED: Example Usage as an async function ---
async def main():
    mcp_url = "http://localhost:8001/mcp"
    injector = MCPToolInjector(mcp_url=mcp_url)
    
    try:
        # Connect and prepare tools
        await injector.connect()
        tool_namespace = injector.get_tool_namespace()
        tool_prompt = injector.get_tools_prompt_string()

        # Initialize agent with the injected tools
        agent = SequentialCodeAgent(
            llm="openai:gpt-4o-mini",
            mcp_tools_namespace=tool_namespace,
            mcp_tools_prompt=tool_prompt,
        )
        
        # New prompt that uses the MCP tools
        final_result = await agent.run(
            "Use the `loud_echo` tool to say hello to the world with 5 exclamation marks."
        )
        print("\n--- AGENT'S FINAL RESPONSE ---")
        print(final_result)

    except ConnectionError as e:
        print(f"Could not run agent: {e}")
        print("Please ensure the MCP server (e.g., echo_server.py) is running.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An unexpected error occurred: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable.")
    finally:
        # Cleanly close the MCP connection
        await injector.close()


if __name__ == "__main__":
    asyncio.run(main())