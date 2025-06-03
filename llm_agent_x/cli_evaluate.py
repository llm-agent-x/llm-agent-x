import argparse
import json
import math
import sys
import time
from os import getenv, environ
from pathlib import Path
from dotenv import load_dotenv
import nltk
import requests
from rich.console import Console
from rich.tree import Tree
from rich.live import Live
from rich.text import Text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any  # Added Any
from enum import Enum
from typing import Literal
import hashlib
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import redis

from llm_agent_x import (
    int_to_base26,
    RecursiveAgent,
    RecursiveAgentOptions,
    TaskLimit,
    TaskObject,
    TaskFailedException,
)
from llm_agent_x.backend import AppendMerger, LLMMerger, AlgorithmicMerger
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from llm_agent_x.console import console, task_tree, live
from llm_agent_x.constants import openai_api_key, openai_base_url
from llm_agent_x.tools.brave_web_search import brave_web_search
from llm_agent_x.llm_manager import model_tree

nltk.download(
    "punkt", quiet=True
)  # Changed from punkt_tab, ensure 'punkt' tokenizer is available

# --- OpenTelemetry Setup ---
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otel_endpoint = getenv("ARIZE_PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
if otel_endpoint and otel_endpoint.lower() != "none":
    exporter = OTLPSpanExporter(endpoint=otel_endpoint)
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))
    print(f"OpenTelemetry configured with endpoint: {otel_endpoint}")
else:
    print(
        "OpenTelemetry endpoint not configured or set to 'none'. Tracing will be local."
    )


# Load environment variables
load_dotenv(".env", override=True)

# Initialize Console
console = Console()

# --- LLM and Search Initialization ---
# Judge LLM (can be configured separately)
judge_llm_instance = None  # To be initialized in main

output_dir = Path(getenv("OUTPUT_DIR", "./output_eval/"))  # Changed default output dir

# --- Caching ---
LANGUAGE = "english"
redis_host = getenv("REDIS_HOST", "localhost")
redis_port = int(getenv("REDIS_PORT", 6379))
redis_db = int(getenv("REDIS_DB", 0))
redis_expiry = int(getenv("REDIS_EXPIRY", 3600))
redis_client: Optional[redis.Redis] = None

outputs = []

try:
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    redis_client.ping()
    print("Redis connection successful.")
except redis.exceptions.ConnectionError as e:
    print(f"Redis connection failed: {e}. Caching disabled.")
    redis_client = None


def exec_python(
    code: str, globals_dict: Optional[Dict] = None, locals_dict: Optional[Dict] = None
) -> Any:
    """
    Execute Python code. For safety, this is a placeholder if you intend to run arbitrary code.
    In a real scenario, use a sandboxed environment.
    """
    console.log(
        f"[bold yellow]Executing Python code (Placeholder - UNSAFE):[/bold yellow]\n{code}"
    )
    # THIS IS A VERY BASIC AND UNSAFE EXECUTION.
    # For production, use a secure sandbox (e.g., Docker container, restricted environment).
    try:
        # For simplicity, using a new dict for globals/locals each time to avoid contamination.
        exec_globals = {}
        exec_locals = {}
        exec(code, exec_globals, exec_locals)
        return {
            "result": "Code executed.",
            "locals": exec_locals,
        }  # Or capture specific output
    except Exception as e:
        return {"error": str(e)}


# --- Callbacks for Rich Tree / Flowchart ---
pre_tasks_executed = None


on_task_executed = None

on_tool_call_executed = None


# --- LLM Judge ---
class JudgeEvaluation(BaseModel):
    score: float = Field(
        description="A numerical score from 1 (worst) to 10 (best) evaluating the response's overall quality."
    )
    reasoning: str = Field(
        description="Detailed reasoning for the score, explaining strengths and weaknesses based on the criteria."
    )
    strengths: List[str] = Field(
        description="Specific positive aspects of the response related to the criteria."
    )
    weaknesses: List[str] = Field(
        description="Specific areas for improvement or failures in the response related to the criteria."
    )
    is_complete: bool = Field(
        description="Does the response fully address all parts of the task as described?"
    )
    is_accurate: bool = Field(
        description="Is the information provided in the response correct and factual (to the best of your knowledge or if verifiable)?"
    )
    is_relevant: bool = Field(
        description="Is the response directly relevant to achieving the stated task?"
    )
    followed_instructions: bool = Field(
        description="Did the agent adhere to specific user instructions, if any were provided?"
    )
    helpfulness: float = Field(
        description="A score from 1 (not helpful) to 10 (very helpful) for how helpful the response is for the user's query."
    )


def evaluate_response_with_llm(
    task_description: str,
    user_instructions: str,
    agent_response: str,
    judge_llm: ChatOpenAI,  # Pass the initialized LLM instance
    pydantic_object_parser: JsonOutputParser,
) -> Dict:

    system_prompt = (
        "You are an impartial and meticulous AI evaluator. Your task is to assess the quality of an AI agent's response "
        "to a given task and user instructions. Provide a numerical score, detailed reasoning, and specific feedback "
        "based on the criteria of Accuracy, Completeness, Relevance, Adherence to Instructions, Clarity, and Helpfulness.\n"
        f"Output your evaluation strictly in the following JSON format:\n{pydantic_object_parser.get_format_instructions()}"
    )

    human_prompt_template = (
        "Please evaluate the following AI agent's performance based on the provided task, user instructions, and the agent's response. "
        "Be critical and fair.\n\n"
        "**Original Task:**\n```\n{task}\n```\n\n"
        "**User Instructions (if any):**\n```\n{u_inst}\n```\n\n"
        "**Agent's Response:**\n```\n{agent_response}\n```\n\n"
        "**Evaluation Criteria to Consider:**\n"
        "- **Score (1-10):** Overall quality. 1=Very Poor, 10=Excellent.\n"
        "- **Reasoning:** Justify your score with specific examples from the response, relating them to the criteria.\n"
        "- **Strengths:** What did the agent do well?\n"
        "- **Weaknesses:** Where did the agent fall short or make errors?\n"
        "- **Completeness:** Did the response fully address all explicit and implicit aspects of the task?\n"
        "- **Accuracy:** Is the information correct? If external knowledge is required and you cannot verify, state so.\n"
        "- **Relevance:** Is the response on-topic and directly useful for accomplishing the task?\n"
        "- **Adherence to Instructions:** Were specific user instructions (e.g., length, tone, format) followed?\n"
        "- **Helpfulness (1-10):** How helpful is this response in achieving the user's goal as stated in the task?\n\n"
        "Provide your evaluation as a JSON object matching the required schema."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=human_prompt_template.format(
                task=task_description,
                u_inst=user_instructions if user_instructions else "None provided.",
                agent_response=(
                    agent_response
                    if agent_response
                    else "No response provided by agent."
                ),
            )
        ),
    ]

    raw_response_content = ""
    try:
        console.log("Invoking Judge LLM...")
        ai_message = judge_llm.invoke(messages)
        raw_response_content = ai_message.content
        # console.log(f"Judge LLM Raw Response: {raw_response_content[:500]}")
        parsed_evaluation = pydantic_object_parser.invoke(ai_message)
        return parsed_evaluation  # This will be a JudgeEvaluation object or dict
    except Exception as e:
        console.log(f"[bold red]Error during LLM Judge evaluation: {e}[/bold red]")
        console.log(f"Judge LLM Raw content on error: {raw_response_content}")
        # Return an error structure matching JudgeEvaluation if possible
        error_eval = JudgeEvaluation(
            score=0,
            reasoning=f"Error during evaluation: {e}. Raw LLM response: {raw_response_content[:500]}",
            strengths=[],
            weaknesses=[f"Evaluation process failed: {e}"],
            is_complete=False,
            is_accurate=False,
            is_relevant=False,
            followed_instructions=False,
            helpfulness=0,
        ).model_dump()  # Convert to dict
        error_eval["error_details"] = str(e)
        return error_eval


TaskType = Literal["research", "search", "basic", "text/reasoning"]


def main():
    global agent_llm  # , judge_llm_instance, live_display, task_tree, flowchart

    parser = argparse.ArgumentParser(description="Run the LLM agent evaluation.")
    parser.add_argument(
        "prompts_file",
        type=str,
        help="Path to the JSON file containing prompts for evaluation.",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="evaluation_results.json",
        help="The output file path for evaluation results.",
    )
    parser.add_argument(
        "--agent_model",
        type=str,
        default=getenv("DEFAULT_LLM", "gpt-4o-mini"),
        help="The name of the LLM for the agent.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=getenv("JUDGE_LLM", "gpt-4o"),
        help="The name of the LLM for the judge.",
    )
    parser.add_argument(
        "--task_limit",
        type=str,
        default="[3,2,2,0]",
        help="Task limits per layer for the agent as a Python list string e.g., '[3,2,2,0]'",
    )
    parser.add_argument(
        "--merger",
        type=str,
        default="ai",
        choices=["ai", "append", "algorithmic"],
        help="Merger type for the agent: 'ai', 'append', or 'algorithmic'.",
    )
    parser.add_argument(
        "--no_live_tree",
        action="store_true",
        help="Disable the real-time Rich tree view during agent runs.",
    )
    # Kept for compatibility with cli.py structure, though not directly used by RecursiveAgent
    parser.add_argument(
        "--default_subtask_type",
        type=str,
        default="basic",
        choices=["research", "search", "basic", "text/reasoning"],
        help="The default task type (currently informational, LLM decides subtask types).",
    )

    # Choose subset of prompts to use, specified as a range
    parser.add_argument(
        "--prompt_range",
        type=str,
        default="-1",
        help="The range of prompts to use, e.g., '1-10'.",
    )

    args = parser.parse_args()

    # --- Initialize/Update LLMs based on args ---
    if args.agent_model != agent_llm.model_name:
        agent_llm = ChatOpenAI(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=args.agent_model,
            temperature=0.5,
        )
        console.log(f"Agent LLM set to: {args.agent_model}")

    judge_llm_instance = ChatOpenAI(
        base_url=openai_base_url,
        api_key=openai_api_key,
        model=args.judge_model,
        temperature=0.2,  # Judge should be more deterministic
    )
    console.log(f"Judge LLM set to: {args.judge_model}")

    # --- Load Prompts ---
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        console.print(f"[bold red]Prompts file not found: {prompts_path}[/bold red]")
        sys.exit(1)

    with prompts_path.open("r") as f:
        try:
            evaluation_prompts = json.load(f)
            if not isinstance(evaluation_prompts, list):
                raise ValueError("Prompts file should contain a JSON array.")
        except json.JSONDecodeError as e:
            console.print(
                f"[bold red]Error decoding JSON from prompts file {prompts_path}: {e}[/bold red]"
            )
            sys.exit(1)
        except ValueError as e:
            console.print(f"[bold red]Error in prompts file structure: {e}[/bold red]")
            sys.exit(1)

    all_evaluations_data = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup Judge Parser ---
    judge_parser = JsonOutputParser(pydantic_object=JudgeEvaluation)

    # --- Main Evaluation Loop ---
    if not args.no_live_tree:
        task_tree = Tree("Agent Execution Evaluation")  # Initialize the main tree
        live_display = Live(
            task_tree,
            console=console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        live_display.start()  # Start live display manually

    # Create subset, if argument==-1, use all prompts
    if args.prompt_range == "-1":
        evaluation_prompts_subset = evaluation_prompts
    else:
        prompt_range = args.prompt_range.split("-")
        start_idx = int(prompt_range[0]) - 1
        end_idx = int(prompt_range[1])
        evaluation_prompts_subset = evaluation_prompts[start_idx:end_idx]

    console.rule("[bold blue]Starting Evaluation[/bold blue]")

    for eval_idx, prompt_data in enumerate(evaluation_prompts_subset):
        prompt_id = prompt_data.get("id", f"eval_{eval_idx:03d}")
        task_description = prompt_data.get("task")
        user_instructions = prompt_data.get(
            "u_inst", prompt_data.get("user_instructions", "")
        )

        if not task_description:
            console.print(
                f"[yellow]Skipping prompt {prompt_id} due to missing 'task' field.[/yellow]"
            )
            continue

        console.rule(
            f"[bold blue]Evaluating Prompt: {prompt_id} ({eval_idx + 1}/{len(evaluation_prompts)})[/bold blue]"
        )
        console.print(f"[cyan]Task:[/cyan] {task_description}")
        if user_instructions:
            console.print(f"[cyan]User Instructions:[/cyan] {user_instructions}")

        flowchart = ["flowchart TD"]
        if task_tree:  # If live tree is active
            task_tree.children.clear()  # Clear children from previous run
            task_tree.label = Text(f"Run: {prompt_id}", style="bold magenta")

        agent_response_text: Optional[str] = None
        agent_run_successful = False
        agent_error_message: Optional[str] = None

        current_eval_data = {
            "prompt_id": prompt_id,
            "task": task_description,
            "user_instructions": user_instructions,
        }

        # Bind fresh tool LLM for each agent instance if tools are stateful or vary
        agent_tool_llm = agent_llm.bind_tools(
            [brave_web_search]
        )  # Add other tools if needed: , exec_python

        with tracer.start_as_current_span(
            f"evaluation_run_{prompt_id}"
        ) as eval_run_span:
            eval_run_span.set_attribute("task", task_description)
            eval_run_span.set_attribute("prompt.id", prompt_id)

            agent_options = RecursiveAgentOptions(
                search_tool=brave_web_search,  # For direct access if needed
                pre_task_executed=(
                    pre_tasks_executed
                    if not args.no_live_tree or args.save_flowcharts
                    else None
                ),
                on_task_executed=(
                    on_task_executed
                    if not args.no_live_tree or args.save_flowcharts
                    else None
                ),
                on_tool_call_executed=(
                    on_tool_call_executed
                    if not args.no_live_tree or args.save_flowcharts
                    else None
                ),
                llm=model_tree,
                tool_llm=agent_tool_llm,
                tools_dict={  # Ensure tools are correctly mapped
                    "brave_web_search": brave_web_search,
                    "web_search": brave_web_search,  # Alias
                    # "exec_python": exec_python, # If you want to enable code execution
                },
                task_limits=TaskLimit.from_array(
                    eval(prompt_data.get("task_limit", args.task_limit))
                ),  # Use eval carefully
                merger={
                    "ai": LLMMerger,
                    "append": AppendMerger,
                    "algorithmic": AlgorithmicMerger,
                }[args.merger],
            )

            agent = RecursiveAgent(
                task=task_description,
                u_inst=user_instructions,
                tracer=tracer,
                tracer_span=eval_run_span,  # Link agent's trace to this evaluation's span
                agent_options=agent_options,
            )

            try:
                console.log("Starting agent run...")
                response_obj = agent.run()
                agent_response_text = str(response_obj)
                outputs.append(
                    {
                        "filename": f"agent_response_{prompt_id}.json",
                        "content": agent_response_text,
                    }
                )
                agent_run_successful = True
                console.log(
                    f"Agent run successful. Response preview: {agent_response_text[:100]}..."
                )
            except TaskFailedException as e:
                agent_error_message = f"TaskFailedException: {e}"
                agent_response_text = f"ERROR: Agent task failed. {e}"
                console.print(
                    f"[bold red]Agent Task Failed: {e}[/bold red]"
                )  # Rich console error
            except Exception as e:
                agent_error_message = f"Unexpected Exception: {e}"
                agent_response_text = (
                    f"ERROR: Agent encountered an unexpected error. {e}"
                )
                console.print(f"[bold red]Agent Unexpected Error: {e}[/bold red]")
                console.print_exception(show_locals=False)

            eval_run_span.set_attribute(
                "agent.response", agent_response_text if agent_response_text else "N/A"
            )
            eval_run_span.set_attribute("agent.run_successful", agent_run_successful)
            if agent_error_message:
                eval_run_span.set_attribute("agent.error_message", agent_error_message)

            # --- LLM Judge Evaluation ---
            console.log("Proceeding to Judge LLM evaluation...")
            judge_eval_result = evaluate_response_with_llm(
                task_description,
                user_instructions,
                agent_response_text,
                judge_llm_instance,
                judge_parser,
            )
            # Ensure it's a dict (evaluate_response_with_llm returns dict from Pydantic model or error dict)
            judge_evaluation_dict = (
                judge_eval_result
                if isinstance(judge_eval_result, dict)
                else judge_eval_result.model_dump()
            )

            console.print(
                f"[bold green]Judge Score for {prompt_id}:[/bold green] {judge_evaluation_dict.get('score')}/10"
            )
            console.print(
                f"[green]Judge Reasoning:[/green] {judge_evaluation_dict.get('reasoning', '')[:250]}..."
            )

            eval_run_span.set_attribute(
                "judge.score", judge_evaluation_dict.get("score", -1)
            )
            eval_run_span.set_attribute(
                "judge.reasoning", judge_evaluation_dict.get("reasoning", "")
            )

            current_eval_data.update(
                {
                    "agent_response": agent_response_text,
                    "agent_run_successful": agent_run_successful,
                    "agent_error_message": agent_error_message,
                    "judge_evaluation": judge_evaluation_dict,
                }
            )
            all_evaluations_data.append(current_eval_data)

    if live_display:
        live_display.stop()  # Stop live display after all evaluations

    # --- Save All Evaluation Results ---
    eval_output_path = output_dir / args.eval_output

    if not eval_output_path.parent.exists():
        eval_output_path.parent.mkdir(parents=True)

    with eval_output_path.open("w") as f_out:
        json.dump(all_evaluations_data, f_out, indent=2)
    console.print(
        f"\n[bold blue]All evaluation results saved to: {eval_output_path}[/bold blue]"
    )

    for output in outputs:
        filename_noprefix = output["filename"]
        output_path = eval_output_path / f"{filename_noprefix}.md"
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        with output_path.open("w") as f_out:
            f_out.write(output["content"])
        console.print(f"\n[bold blue]Output saved to: {output_path}[/bold blue]")
    # --- Print Summary ---
    if all_evaluations_data:
        num_evaluated = len(all_evaluations_data)
        successful_agent_runs = sum(
            1 for r in all_evaluations_data if r["agent_run_successful"]
        )
        avg_score = (
            sum(r["judge_evaluation"].get("score", 0) for r in all_evaluations_data)
            / num_evaluated
            if num_evaluated > 0
            else 0
        )
        avg_helpfulness = (
            sum(
                r["judge_evaluation"].get("helpfulness", 0)
                for r in all_evaluations_data
            )
            / num_evaluated
            if num_evaluated > 0
            else 0
        )

        console.rule("[bold green]Evaluation Summary[/bold green]")
        console.print(f"Total Prompts Evaluated: {num_evaluated}")
        if num_evaluated:
            percentage = (successful_agent_runs / num_evaluated) * 100
            console.print(
                f"Successful Agent Runs: {successful_agent_runs} ({percentage:.2f}%)"
            )
        else:
            console.print(f"Successful Agent Runs: {successful_agent_runs} (0.00%)")
        console.print(f"Average Judge Score: {avg_score:.2f}/10")
        console.print(f"Average Judge Helpfulness Score: {avg_helpfulness:.2f}/10")


if __name__ == "__main__":
    main()
