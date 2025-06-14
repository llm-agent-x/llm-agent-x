import io
import subprocess
import sys
import os
import pickle
from typing import List
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import base64
import traceback
import ast
import astunparse

app = FastAPI()
globals_ = {}
locals_ = {}

# Optionally require an API token depending on the environment
requires_api_token = (
    os.getenv("PYTHON_SANDBOX_API_TOKEN_REQUIRED", "false").lower() == "true"
)
api_token = os.getenv("PYTHON_SANDBOX_API_TOKEN")

if requires_api_token and not api_token:
    raise RuntimeError(
        "PYTHON_SANDBOX_API_TOKEN_REQUIRED is set to true, but PYTHON_SANDBOX_API_TOKEN is not set"
    )
if not requires_api_token and api_token:
    raise RuntimeError(
        "PYTHON_SANDBOX_API_TOKEN_REQUIRED is set to false, but PYTHON_SANDBOX_API_TOKEN is set"
    )
if not requires_api_token:
    print(
        "\033[93mWarning:\033[0m PYTHON_SANDBOX_API_TOKEN_REQUIRED is set to false. No authentication will be required."
    )


class CodeRequest(BaseModel):
    encoded_code: str


class FileUploadRequest(BaseModel):
    encoded_file: str
    filename: str


@app.post("/upload_file")
async def upload_file(
    request: FileUploadRequest, credentials: HTTPAuthorizationCredentials = Depends()
):
    """
    Uploads a file to the sandbox's workspace.
    """
    if requires_api_token and credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

    if requires_api_token and credentials.credentials != api_token:
        raise HTTPException(status_code=403, detail="Invalid API token")

    encoded_file = request.encoded_file
    if not encoded_file:
        return {"message": "Missing 'encoded_file' in JSON payload"}

    filename = request.filename
    if not filename:
        return {"message": "Missing 'filename' in JSON payload"}

    # Decode the base64 encoded file
    try:
        file = base64.b64decode(encoded_file)
    except Exception as e:
        return {"message": f"Invalid base64 encoding: {str(e)}"}

    # Save the file to the sandbox's workspace
    with open(
        os.path.join(os.getenv("PYTHON_SANDBOX_WORKSPACE_DIR"), filename), "wb"
    ) as f:
        f.write(file)

    return {"message": f"Uploaded {filename} to the sandbox's workspace"}


class FileUploadRequestWithLocalName(BaseModel):
    encoded_file: str
    filename: str
    local_name: str


@app.post("/upload_pickle")
async def upload_pickle(
    request: FileUploadRequestWithLocalName,
    credentials: HTTPAuthorizationCredentials = Depends(),
):
    """
    Uploads a pickle file to the sandbox's workspace and loads it into a local variable.
    """
    if requires_api_token and credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

    if requires_api_token and credentials.credentials != api_token:
        raise HTTPException(status_code=403, detail="Invalid API token")

    encoded_file = request.encoded_file
    if not encoded_file:
        return {"message": "Missing 'encoded_file' in JSON payload"}

    filename = request.filename
    if not filename:
        return {"message": "Missing 'filename' in JSON payload"}

    # Decode the base64 encoded file
    try:
        file = base64.b64decode(encoded_file)
    except Exception as e:
        return {"message": f"Invalid base64 encoding: {str(e)}"}

    # Save the file to the sandbox's workspace
    with open(
        os.path.join(os.getenv("PYTHON_SANDBOX_WORKSPACE_DIR"), filename), "wb"
    ) as f:
        f.write(file)

    # Load the pickle file into a local variable
    with open(
        os.path.join(os.getenv("PYTHON_SANDBOX_WORKSPACE_DIR"), filename), "rb"
    ) as f:
        local_name = request.local_name
        if local_name:
            loaded_pickle = pickle.load(f)
            locals()[local_name] = loaded_pickle
        else:
            # If no name is provided, don't load into locals
            return {
                "message": f"Uploaded {filename} to the sandbox's workspace",
            }

    return {
        "message": f"Uploaded {filename} to the sandbox's workspace and loaded it as {local_name}",
        "loaded_pickle": loaded_pickle,
    }


@app.post("/execute")
async def execute_code(
    request: CodeRequest, credentials: HTTPAuthorizationCredentials = Depends()
):
    """
    Executes Python code within the sandbox and returns stdout/stderr and the last expression value.
    """
    if requires_api_token and credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

    if requires_api_token and credentials.credentials != api_token:
        raise HTTPException(status_code=403, detail="Invalid API token")

    encoded_code = request.encoded_code
    if not encoded_code:
        return {
            "stdout": "",
            "stderr": "",
            "result": "",
            "message": "Missing 'encoded_code' in JSON payload",
        }

    # Decode the base64 encoded code
    try:
        code = base64.b64decode(encoded_code).decode("utf-8")
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "result": "",
            "message": f"Invalid base64 encoding: {str(e)}",
        }

    # Prepare globals and locals dictionaries
    execution_globals = {}
    execution_locals = {}

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = io.StringIO()
    sys.stderr = captured_stderr = io.StringIO()

    last_expression_value = None
    modified_code = code
    try:
        # Parse the code to determine if it's an expression or statement
        module = ast.parse(code)

        # Modify the AST to wrap standalone expressions in print()
        for i, node in enumerate(module.body):
            if isinstance(node, ast.Expr):
                # Create a print statement node
                print_value = ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[node.value],
                    keywords=[],
                )
                # Replace the expression with the print statement
                module.body[i] = ast.Expr(value=print_value)
        # Unparse AST to create a new string of code
        modified_code = astunparse.unparse(module)

        # Execute the code
        exec(modified_code, execution_globals, execution_locals)

        # Get the last expression value (if any)
        last_expression_value = execution_locals.get("_")

        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        result_val = (
            str(last_expression_value) if last_expression_value is not None else ""
        )

        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "result": result_val,
            "message": "Code executed successfully.",
        }
    except Exception as e:
        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        traceback_str = traceback.format_exc()
        error_message = f"Error during execution: {str(e)}"

        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "result": "",
            "message": error_message,
            "trace": traceback_str,
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@app.post("/install")
async def install_packages(
    packages: List[str],
    index_url: str = None,
    credentials: HTTPAuthorizationCredentials = Depends(),
):
    """
    Install packages to the sandbox environment.

    Parameters:
    packages (list): List of packages to install, optionally specifying versions.
    index_url (str, optional): URL of the package index to use. Defaults to None.
    """
    if requires_api_token and credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

    if requires_api_token and credentials.credentials != api_token:
        raise HTTPException(status_code=403, detail="Invalid API token")

    if index_url:
        cmd = ["pip", "install", "--index-url", index_url] + packages
    else:
        cmd = ["pip", "install"] + packages

    result = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500, detail=f"Failed to install packages: {result.stderr}"
        )
    return {"message": "Packages installed successfully."}
