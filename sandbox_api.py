import os
import cloudpickle
import traceback
import sys
from io import StringIO
from flask import Flask, request, jsonify

app = Flask(__name__)

# Directory for user-uploaded files
WORKSPACE_DIR = "/workspace"
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# Global dictionary to store loaded pickle objects
# Keys will be file paths, values will be the unpickled objects
LOADED_PICKLES = {}

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Uploads a file to the sandbox's /workspace directory.
    Expects a POST request with 'multipart/form-data'.
    The file should be part of the request under the key 'file'.
    Returns:
        JSON: Success message with filename or error message.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = os.path.join(WORKSPACE_DIR, file.filename)
        # Ensure the filename is within the workspace_dir to prevent directory traversal
        if not os.path.abspath(filename).startswith(os.path.abspath(WORKSPACE_DIR)):
             return jsonify({"error": "Invalid file path"}), 400
        try:
            file.save(filename)
            return jsonify({"message": f"File '{file.filename}' uploaded successfully to {filename}"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/load_pickle', methods=['POST'])
def load_pickle_route():
    """
    Loads a cloudpickle file from the /workspace directory into memory.
    Expects a POST request with a JSON payload: {"file_path": "path/to/your.pkl"}
    The 'file_path' is relative to the /workspace directory.
    The loaded object is stored in a global dictionary `LOADED_PICKLES`
    keyed by its 'file_path'.
    Returns:
        JSON: Success or error message.
    """
    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({"error": "Missing 'file_path' in JSON payload"}), 400

    pickle_file_path = data['file_path']
    # Construct the full path within the workspace
    full_pickle_path = os.path.join(WORKSPACE_DIR, pickle_file_path)

    # Security check: ensure the path is within WORKSPACE_DIR
    if not os.path.abspath(full_pickle_path).startswith(os.path.abspath(WORKSPACE_DIR)):
        return jsonify({"error": "Access denied: Path is outside the workspace."}), 403

    if not os.path.exists(full_pickle_path):
        return jsonify({"error": f"Pickle file not found: {full_pickle_path}"}), 404

    try:
        with open(full_pickle_path, 'rb') as f:
            obj = cloudpickle.load(f)
        # Store the loaded object using its workspace-relative path as key
        LOADED_PICKLES[pickle_file_path] = obj
        return jsonify({"message": f"Object from '{pickle_file_path}' loaded successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Error loading pickle: {str(e)}", "trace": traceback.format_exc()}), 500

@app.route('/execute', methods=['POST'])
def execute_code():
    """
    Executes Python code within the sandbox.
    Expects a POST request with a JSON payload: {"code": "your python code string"}
    The code can access objects loaded via /load_pickle through the `LOADED_PICKLES` dict.
    Files uploaded to /workspace can also be accessed by the code if it knows their path.
    Returns:
        JSON: Contains 'stdout', 'stderr', and 'error' (if any) from the execution.
    """
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({"error": "Missing 'code' in JSON payload"}), 400

    code_to_execute = data['code']

    # Prepare a globals dictionary for exec, including loaded pickles
    # This allows the executed code to access objects loaded by /load_pickle
    # E.g., if 'my_data.pkl' was loaded, it's available as LOADED_PICKLES['my_data.pkl']
    execution_globals = {
        "LOADED_PICKLES": LOADED_PICKLES,
        **globals() # Includes other globals from this script if necessary
    }

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = StringIO()
    sys.stderr = captured_stderr = StringIO()

    try:
        exec(code_to_execute, execution_globals)
        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        return jsonify({
            "stdout": stdout_val,
            "stderr": stderr_val,
            "message": "Code executed successfully."
        }), 200
    except Exception as e:
        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        return jsonify({
            "stdout": stdout_val,
            "stderr": stderr_val,
            "error": f"Error during execution: {str(e)}",
            "trace": traceback.format_exc()
        }), 500
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True for development
