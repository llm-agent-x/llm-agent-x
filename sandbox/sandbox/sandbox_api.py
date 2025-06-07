import io
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import traceback
import ast
import astunparse

app = FastAPI()

class CodeRequest(BaseModel):
    encoded_code: str

@app.post("/execute")
async def execute_code(request: CodeRequest):
    """
    Executes Python code within the sandbox and returns stdout/stderr and the last expression value.
    """
    encoded_code = request.encoded_code
    if not encoded_code:
        return {"stdout":"", "stderr":"", "result":"", "message": "Missing 'encoded_code' in JSON payload"}

    # Decode the base64 encoded code
    try:
        code = base64.b64decode(encoded_code).decode('utf-8')
    except Exception as e:
        return {"stdout":"", "stderr":"", "result":"", "message": f"Invalid base64 encoding: {str(e)}"}

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
                    func=ast.Name(id='print', ctx=ast.Load()),
                    args=[node.value],
                    keywords=[]
                )
                # Replace the expression with the print statement
                module.body[i] = ast.Expr(value=print_value)
        #Unparse AST to create a new string of code
        modified_code = astunparse.unparse(module)

        # Execute the code
        exec(modified_code, execution_globals, execution_locals)

        # Get the last expression value (if any)
        last_expression_value = execution_locals.get('_')

        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        result_val = str(last_expression_value) if last_expression_value is not None else ""


        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "result": result_val,
            "message": "Code executed successfully."
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
            "trace": traceback_str
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

@app.get("/health_check")
async def health_check():
    return {"status": "OK"}