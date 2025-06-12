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

# --- Asynchronous Python Executor (Unchanged) ---
async def aexec_python_local(code: str, globals: Dict = None, locals: Dict = None) -> Dict[str, Any]:
    from io import StringIO
    import sys
    if globals is None: globals = {}
    if locals is None: locals = {}
    globals['asyncio'] = asyncio
    old_stdout, old_stderr = sys.stdout, sys.stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = redirected_stdout = StringIO()
    sys.stderr = redirected_stderr = StringIO()
    wrapped_code = "async def __aexec_wrapper__():\n" + "".join(f"    {line}\n" for line in code.splitlines())
    try:
        exec(wrapped_code, globals, locals)
        result = await locals['__aexec_wrapper__']()
        if result is not None:
            print(result)
    except Exception:
        tb = traceback.format_exc()
        redirected_stderr.write(tb)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return {"stdout": redirected_stdout.getvalue(), "stderr": redirected_stderr.getvalue()}

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
        return self # Return self to be used in the 'as' part of 'async with'

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection to the MCP server."""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
        if self._streams:
            await self._streams.__aexit__(exc_type, exc_val, exc_tb)
        ic("MCP connection closed.")

    def _create_callable_for_tool(self, tool: types.Tool) -> Callable:
        """Dynamically creates an async function to wrap an MCP tool call."""
        async def mcp_tool_wrapper(**kwargs):
            # ic(f"Calling MCP tool '{tool.name}' with arguments: {kwargs}")
            tool_call_result = await self._session.call_tool(tool.name, arguments=kwargs)
            # The actual return value is in tool_call_result.content
            # It's a list of blocks; we'll assume the first is what we want.
            if tool_call_result.content and tool_call_result.content[0].type == "text":
                txt = f"Tool '{tool.name}' returned text: {tool_call_result.content[0].text}"
                ic(txt)
                return tool_call_result.content[0].text
            return tool_call_result.content

        # Create a user-friendly docstring for the LLM from the OpenAPI schema
        schema = tool.inputSchema
        arg_docs_list = []
        if "properties" in schema:
             for name, prop in schema.get("properties", {}).items():
                arg_docs_list.append(f"        - {name} ({prop.get('type', 'any')}): {prop.get('description', '')}")
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
        prompt_str += "You MUST use `await` when calling these functions (e.g., `result = await get_weather(city='London')`).\n\n"
        for tool in self._tools:
            # Use the generated docstring for a richer prompt
            callable_tool = self._create_callable_for_tool(tool)
            prompt_str += f"- `async def {tool.name}(...)`:\n"
            doc = "\n".join([f"  {line}" for line in callable_tool.__doc__.strip().split('\n')])
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
    code: Annotated[str, AfterValidator(check_and_return_code)] = Field(
        description="A string containing valid Python code to be executed."
    )

# --- Async SequentialCodeAgent (Unchanged) ---
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
            "1. You can write and execute Python code. To do so, output a JSON object like: `{\"code\": \"print('hello')\"}`.\n"
            "2. The Python environment is STATEFUL. Variables persist.\n"
            "3. Use `print()` to see the result of any operation. The return value of `await` calls is automatically printed.\n"
            "4. When you have the final answer, respond with a plain string."
        )
        return base_prompt + mcp_tools_prompt

    async def _execute_code_in_sandbox(self, code: str) -> Dict[str, Any]:
        ic("Passing code to aexec_python_local with stateful namespace")
        return await aexec_python_local(code=code, globals=self.namespace_globals, locals=self.namespace_locals)

    async def run(self, prompt: str):
        current_prompt = f"Here is your task: \n\n\"{prompt}\""
        for i in range(self.max_turns):
            ic(f"--- Turn {i+1}/{self.max_turns} ---")
            response = await self.agent.run(current_prompt, message_history=self.history)
            self.history = response.all_messages()
            if isinstance(response.output, Code):
                code_to_run = response.output.code
                ic(f"Executing code:\n---\n{code_to_run}\n---")
                python_result = await self._execute_code_in_sandbox(code=code_to_run)
                ic(python_result)
                stdout = python_result.get('stdout', '').strip()
                stderr = python_result.get('stderr', '').strip()
                current_prompt = "The code execution produced the following output. Continue with your task."
                if stdout: current_prompt += f"\n\nSTDOUT:\n```\n{stdout}\n```"
                if stderr: current_prompt += f"\n\nSTDERR:\n```\n{stderr}\n```"
                if not stdout and not stderr: current_prompt += "\n\nNOTE: The code ran without error and produced no output."
            elif isinstance(response.output, str):
                ic(f"Final answer received: {response.output}")
                return response.output
            else:
                error_message = f"Error: Unexpected response type from agent: {type(response.output)}"
                ic(error_message)
                return error_message
        print("Agent reached maximum turns without providing a final answer.")
        return None

# --- REFACTORED: Main usage with 'async with' ---
async def main():
    mcp_url = "http://localhost:8001/mcp"
    
    try:
        # Use the injector as a context manager
        async with MCPToolInjector(mcp_url=mcp_url) as injector:
            # Prepare tools and prompt inside the 'with' block
            tool_namespace = injector.get_tool_namespace()
            tool_prompt = injector.get_tools_prompt_string()

            agent = SequentialCodeAgent(
                llm="openai:gpt-4o-mini",
                mcp_tools_namespace=tool_namespace,
                mcp_tools_prompt=tool_prompt,
            )
            
            # Use a prompt that matches an available tool
            final_result = await agent.run(
                "Find out the weather in Tokyo using the available tool."
            )
            print("\n--- AGENT'S FINAL RESPONSE ---")
            print(final_result)

    except ConnectionError as e:
        print(f"\n[ERROR] Could not run agent: {e}")
        print("Please ensure the MCP server (e.g., the reference 'examples/python/tools_server.py') is running.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable.")

if __name__ == "__main__":
    # To run this, you need an MCP server running.
    # For example, from the mcp-sdk-python repo:
    # python examples/python/tools_server.py
    asyncio.run(main())