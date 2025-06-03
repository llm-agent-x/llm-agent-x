import io
import sys
from fastapi import FastAPI, File, UploadFile
import os
import cloudpickle
import traceback
from typing import Dict

app = FastAPI()

# Directory for user-uploaded files
WORKSPACE_DIR = "/workspace"
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# Global dictionary to store loaded pickle objects
# Keys will be file paths, values will be the unpickled objects
LOADED_PICKLES: Dict[str, object] = {}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file to the sandbox's /workspace directory.
    Returns:
        JSON: Success message with filename or error message.
    """
    if not file:
        return {"error": "No file part"}, 400
    if file.filename == "":
        return {"error": "No selected file"}, 400
    filename = os.path.join(WORKSPACE_DIR, file.filename)
    # Ensure the filename is within the workspace_dir to prevent directory traversal
    if not os.path.abspath(filename).startswith(os.path.abspath(WORKSPACE_DIR)):
        return {"error": "Invalid file path"}, 400
    try:
        contents = await file.read()
        with open(filename, "wb") as f:
            f.write(contents)
        return {
            "message": f"File '{file.filename}' uploaded successfully to {filename}"
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


@app.post("/load_pickle")
async def load_pickle(file_path: str):
    """
    Loads a cloudpickle file from the /workspace directory into memory.
    The 'file_path' is relative to the /workspace directory.
    The loaded object is stored in a global dictionary `LOADED_PICKLES`
    keyed by its 'file_path'.
    Returns:
        JSON: Success or error message.
    """
    pickle_file_path = file_path
    # Construct the full path within the workspace
    full_pickle_path = os.path.join(WORKSPACE_DIR, pickle_file_path)

    # Security check: ensure the path is within WORKSPACE_DIR
    if not os.path.abspath(full_pickle_path).startswith(os.path.abspath(WORKSPACE_DIR)):
        return {"error": "Access denied: Path is outside the workspace."}, 403

    if not os.path.exists(full_pickle_path):
        return {"error": f"Pickle file not found: {full_pickle_path}"}, 404

    try:
        with open(full_pickle_path, "rb") as f:
            obj = cloudpickle.load(f)
        # Store the loaded object using its workspace-relative path as key
        LOADED_PICKLES[pickle_file_path] = obj
        return {
            "message": f"Object from '{pickle_file_path}' loaded successfully."
        }, 200
    except Exception as e:
        return {
            "error": f"Error loading pickle: {str(e)}",
            "trace": traceback.format_exc(),
        }, 500


@app.post("/execute")
async def execute_code(code: str):
    """
    Executes Python code within the sandbox.
    The code can access objects loaded via /load_pickle through the `LOADED_PICKLES` dict.
    Files uploaded to /workspace can also be accessed by the code if it knows their path.
    Returns:
        JSON: Contains 'stdout', 'stderr', and 'error' (if any) from the execution.
        You can use the preloaded `final_response` function to construct a custom response,
        which will, if called, overwrite the captured stdout and stderr.
    """
    if not code:
        return {"error": "Missing 'code' in JSON payload"}, 400

    # Prepare a globals dictionary for exec, including loaded pickles
    # This allows the executed code to access objects loaded by /load_pickle
    # E.g., if 'my_data.pkl' was loaded, it's available as LOADED_PICKLES['my_data.pkl']
    execution_globals = {
        "LOADED_PICKLES": LOADED_PICKLES,
        **globals(),  # Includes other globals from this script if necessary
        "final_response": final_response,
    }

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = io.StringIO()
    sys.stderr = captured_stderr = io.StringIO()

    try:
        exec(code, execution_globals)
        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        response = {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "message": "Code executed successfully.",
        }
        if "final_response" in execution_globals:
            # response.update(execution_globals["final_response"])
            response = execution_globals["final_response"]
        return response, 200
    except Exception as e:
        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "error": f"Error during execution: {str(e)}",
            "trace": traceback.format_exc(),
        }, 500
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def final_response(stdout=None, stderr=None, error=None, message=None, trace=None):
    """
    Returns a dict that can be used to construct the final response from the /execute endpoint.
    Note that this dict is merged with the captured stdout and stderr, so if you want to ignore
    the captured output, set stdout and stderr to None.
    """
    return {
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "message": message,
        "trace": trace,
    }
