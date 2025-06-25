import io
import sys
import requests
import os
import json
import base64
import cloudpickle
from llm_agent_x.constants import SANDBOX_API_URL

# Configuration for the Dockerized sandbox API


def install_packages(packages, index_url=None):
    # Install packages using the Dockerized sandbox API
    response = requests.post(
        f"{SANDBOX_API_URL}/install",
        json={"packages": packages, "index_url": index_url},
    )
    return response.json()


from typing import Dict, List, Optional
import requests
import os
import json
import base64
import cloudpickle
from llm_agent_x.constants import SANDBOX_API_URL


def exec_python_factory(use_docker_sandbox: bool = True):
    def exec_python(
        code: str,
        files_to_upload: Optional[List[str]] = None,
        cloud_pickle_files_to_load: Optional[List[str]] = None,
        globals: Optional[Dict] = None,
        locals: Optional[Dict] = None,
        packages: Optional[List[str]] = None,
        packages_index_url: Optional[str] = None,
    ) -> Optional[Dict]:
        if use_docker_sandbox:
            if not SANDBOX_API_URL:
                return {
                    "stdout": "",
                    "stderr": "PYTHON_SANDBOX_API_URL environment variable is not set.",
                    "error": "Configuration error",
                }

            if packages:
                install_packages(packages, packages_index_url)

            results = {"stdout": "", "stderr": "", "error": None}

            if files_to_upload:
                for file_path in files_to_upload:
                    try:
                        with open(file_path, "rb") as f:
                            file_name = os.path.basename(file_path)
                            response = requests.post(
                                f"{SANDBOX_API_URL}/upload",
                                files={"file": (file_name, f)},
                            )
                            response.raise_for_status()
                    except FileNotFoundError:
                        results[
                            "stderr"
                        ] += f"Error: File not found for upload: {file_path}\n"
                        results["error"] = "File upload error"
                        return results
                    except requests.exceptions.RequestException as e:
                        results["stderr"] += f"Error uploading {file_path}: {e}\n"
                        results["error"] = "File upload error"
                        return results

            if cloud_pickle_files_to_load:
                for cp_file_path in cloud_pickle_files_to_load:
                    try:
                        response = requests.post(
                            f"{SANDBOX_API_URL}/load_pickle",
                            json={"file_path": cp_file_path},
                        )
                        response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        results[
                            "stderr"
                        ] += f"Error loading cloudpickle file {cp_file_path}: {e}\n"
                        try:
                            error_detail = response.json()
                            results[
                                "stderr"
                            ] += f"Sandbox response: {error_detail.get('error', '')} - {error_detail.get('trace', '')}\n"
                        except ValueError:
                            results["stderr"] += f"Sandbox response: {response.text}\n"
                        results["error"] = "Cloudpickle load error"
                        return results

            try:
                code_b64 = base64.b64encode(code.encode()).decode()
                response = requests.post(
                    f"{SANDBOX_API_URL}/execute", json={"encoded_code": code_b64}
                )
                response.raise_for_status()
                exec_result = response.json()
                results["stdout"] = exec_result.get("stdout", "")
                results["stderr"] += exec_result.get("stderr", "")
                if exec_result.get("error"):
                    results["error"] = exec_result.get("error")
                    results[
                        "stderr"
                    ] += f"Execution error from sandbox: {exec_result.get('error')}\n"
                    if exec_result.get("trace"):
                        results[
                            "stderr"
                        ] += f"Sandbox Trace: {exec_result.get('trace')}\n"

            except requests.exceptions.RequestException as e:
                results["stderr"] += f"Error executing code in sandbox: {e}\n"
                try:
                    error_detail = response.json()
                    results[
                        "stderr"
                    ] += f"Sandbox response: {error_detail.get('error', '')} - {error_detail.get('trace', '')}\n"
                except ValueError:
                    results["stderr"] += f"Sandbox response: {response.text}\n"
                results["error"] = "Code execution error"

            results.update(
                {
                    "instructions": "Use the outputs or errors to respond to the query. If it was successful and you got the information you need, relay it to the user."
                }
            )
            return results

        else:
            try:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = captured_stdout = io.StringIO()
                sys.stderr = captured_stderr = io.StringIO()

                exec(
                    code,
                    globals if globals is not None else {},
                    locals if locals is not None else {},
                )

                stdout = captured_stdout.getvalue()
                stderr = captured_stderr.getvalue()

                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": None,
                }
            except Exception as e:
                return {
                    "stdout": "",
                    "stderr": f"[Local execution error: {str(e)}]",
                    "error": str(e),
                }

    return exec_python


exec_python = exec_python_factory(use_docker_sandbox=True)
exec_python_local = exec_python_factory(use_docker_sandbox=False)
