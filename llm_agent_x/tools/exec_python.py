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


def exec_python(
    code,
    files_to_upload=None,
    cloud_pickle_files_to_load=None,
    globals=None,
    locals=None,
    packages=None,
    packages_index_url=None,
):
    """
    Execute the given Python code.

    Parameters:
    code (str): The Python code to be executed.
    globals (dict, optional): A dictionary of global variables for local execution. Defaults to None.
                              Not used if use_docker_sandbox is True.
    locals (dict, optional): A dictionary of local variables for local execution. Defaults to None.
                             Not used if use_docker_sandbox is True.
    packages (list, optional): List of additional packages to be installed in the sandbox environment.
    packages_index_url (str, optional): URL of the package index to use for package installation.

    For example:
    exec_python("x = 10; y = 20; z = x + y; print(z)")

    Returns:
    dict or None: If using Docker sandbox, returns a dictionary with 'stdout', 'stderr', and 'error' (if any).
                  If local execution, returns None. (Note: local exec() doesn't directly return stdout/stderr,
                  this might need further refinement if local output capture is critical). Anything you need,
                  you can print it to the console, but don't print too much, because there is a 512 char limit.
    """
    use_docker_sandbox = True
    if use_docker_sandbox:
        # Ensure the sandbox URL is configured
        if not SANDBOX_API_URL:
            return {
                "stdout": "",
                "stderr": "PYTHON_SANDBOX_API_URL environment variable is not set.",
                "error": "Configuration error",
            }

        # Install packages if any
        if packages:
            install_packages(packages, packages_index_url)

        # Prepare the payload
        results = {"stdout": "", "stderr": "", "error": None}

        # 1. Upload files
        if files_to_upload:
            for file_path in files_to_upload:
                try:
                    with open(file_path, "rb") as f:
                        file_name = os.path.basename(file_path)
                        response = requests.post(
                            f"{SANDBOX_API_URL}/upload", files={"file": (file_name, f)}
                        )
                        response.raise_for_status()
                        # Optionally log response.json().get("message")
                except FileNotFoundError:
                    results[
                        "stderr"
                    ] += f"Error: File not found for upload: {file_path}\n"
                    results["error"] = "File upload error"
                    return results  # Stop if a file can't be uploaded
                except requests.exceptions.RequestException as e:
                    results["stderr"] += f"Error uploading {file_path}: {e}\n"
                    results["error"] = "File upload error"
                    return results

        # 2. Load cloudpickle files
        if cloud_pickle_files_to_load:
            for cp_file_path in cloud_pickle_files_to_load:
                try:
                    # cp_file_path is relative to the workspace, e.g., "my_data.pkl"
                    response = requests.post(
                        f"{SANDBOX_API_URL}/load_pickle",
                        json={"file_path": cp_file_path},
                    )
                    response.raise_for_status()
                    # Optionally log response.json().get("message")
                except requests.exceptions.RequestException as e:
                    results[
                        "stderr"
                    ] += f"Error loading cloudpickle file {cp_file_path}: {e}\n"
                    # Attempt to get more details from the sandbox response if available
                    try:
                        error_detail = response.json()
                        results[
                            "stderr"
                        ] += f"Sandbox response: {error_detail.get('error', '')} - {error_detail.get('trace', '')}\n"
                    except ValueError:  # If response is not JSON
                        results["stderr"] += f"Sandbox response: {response.text}\n"
                    results["error"] = "Cloudpickle load error"
                    return results

        # 3. Execute code
        try:
            # Encode code in base64 before sending it to the server
            code_b64 = base64.b64encode(code.encode()).decode()
            from icecream import ic

            ic(code_b64)

            response = requests.post(
                f"{SANDBOX_API_URL}/execute", json={"encoded_code": code_b64}
            )
            response.raise_for_status()
            exec_result = response.json()
            results["stdout"] = exec_result.get("stdout", "")
            results["stderr"] += exec_result.get(
                "stderr", ""
            )  # Append to any previous stderr
            if exec_result.get("error"):
                results["error"] = exec_result.get("error")
                results[
                    "stderr"
                ] += f"Execution error from sandbox: {exec_result.get('error')}\n"
                if exec_result.get("trace"):
                    results["stderr"] += f"Sandbox Trace: {exec_result.get('trace')}\n"

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
        # Original local execution (consider capturing stdout/stderr if needed for consistency)
        # For simplicity, this part remains as is, but for production, you might want
        # to use subprocess or other methods to capture output from local exec as well.
        try:
            exec(
                code,
                globals if globals is not None else {},
                locals if locals is not None else {},
            )
            # Local execution doesn't directly return stdout/stderr or errors in this simple form
            return {
                "stdout": "[Local execution - stdout not captured]",
                "stderr": "[Local execution - stderr not captured]",
                "error": None,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"[Local execution error: {str(e)}]",
                "error": str(e),
            }
