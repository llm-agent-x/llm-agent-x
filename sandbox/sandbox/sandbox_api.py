import io
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import traceback

app = FastAPI()

class CodeRequest(BaseModel):
    encoded_code: str

@app.post("/execute")
async def execute_code(request: CodeRequest):
    """
    Executes Python code within the sandbox.
    """
    encoded_code = request.encoded_code
    if not encoded_code:
        return {"error": "Missing 'encoded_code' in JSON payload"}, 400

    # Decode the base64 encoded code
    try:
        code = base64.b64decode(encoded_code).decode('utf-8')
    except Exception as e:
        return {"error": f"Invalid base64 encoding: {str(e)}"}, 400

    # Prepare a globals dictionary for exec
    execution_globals = {
        **globals(),
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
        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "message": "Code executed successfully.",
        }
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

@app.get("/health_check")
async def health_check():
    return {"status": "OK"}