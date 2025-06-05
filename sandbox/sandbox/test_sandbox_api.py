import os
import sys
import unittest
import tempfile
import cloudpickle
import json

# Add the directory containing sandbox_api to sys.path
# Assuming sandbox_api.py is in the same directory as this test script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import the app and other necessary components from sandbox_api
# We need to ensure WORKSPACE_DIR is set up for testing before app is imported fully
# So we'll do a bit of a controlled import.

# Create a temporary workspace for testing
temp_workspace = tempfile.TemporaryDirectory()
os.environ["PYTHON_SANDBOX_WORKSPACE_DIR"] = temp_workspace.name

# It's tricky to modify sandbox_api.WORKSPACE_DIR after import.
# A common pattern for configurable Flask apps is to have a create_app function.
# Since sandbox_api.py is simple, we'll manipulate its global WORKSPACE_DIR for testing,
# or ideally, refactor sandbox_api.py to use app.config for WORKSPACE_DIR.

# For now, let's assume we can set WORKSPACE_DIR in sandbox_api.py before app.testing = True
# This is a bit of a hack. A better way would be to refactor sandbox_api.py
# to initialize WORKSPACE_DIR from Flask app.config or an environment variable that can be set here.

# Let's try to import the app after setting an environment variable that sandbox_api.py could use.
# Modifying sandbox_api.py to respect an env var for WORKSPACE_DIR would be cleaner.
# For this subtask, we proceed with the assumption that sandbox_api.py uses /workspace
# and we will mock/patch its WORKSPACE_DIR or use the real one if permissions allow.

# Given the current structure of sandbox_api.py, we will directly import `app`
# and then monkeypatch its WORKSPACE_DIR for testing.
from sandbox.sandbox.sandbox_api import app, LOADED_PICKLES


class SandboxAPITestCase(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

        # Create a temporary workspace for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_workspace_dir = app.config.get(
            "WORKSPACE_DIR", "/workspace"
        )  # Assuming it might be in config

        # Monkeypatch WORKSPACE_DIR in sandbox_api module
        # This requires sandbox_api.py to be written to respect this variable.
        # If sandbox_api.py hardcodes WORKSPACE_DIR = "/workspace", this won't redirect file operations
        # unless the tests are run inside a container or with permissions to write to /workspace.
        # For robust testing, sandbox_api.py should take WORKSPACE_DIR from app.config.
        # Let's assume we are testing the provided sandbox_api.py as is.
        # We will save files to a temporary location and try to make the API use it.

        # For the purpose of this test, we'll assume WORKSPACE_DIR in sandbox_api.py
        # is the one defined there. We will create files in a temp_dir and then
        # tell the API to load them from a path *as if* they were in its WORKSPACE_DIR.
        # This is not ideal. The API should really use a configurable WORKSPACE_DIR.

        # Reset loaded pickles before each test
        LOADED_PICKLES.clear()

        # Create a temporary directory that will simulate the app's WORKSPACE_DIR for uploads
        self.test_workspace = tempfile.TemporaryDirectory()
        # Monkey patch the WORKSPACE_DIR in the imported app module for the duration of the test
        self.actual_sandbox_api_workspace_dir = app.WORKSPACE_DIR  # store original
        app.WORKSPACE_DIR = self.test_workspace.name

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()
        self.test_workspace.cleanup()
        # Restore original WORKSPACE_DIR
        app.WORKSPACE_DIR = self.actual_sandbox_api_workspace_dir

    def test_0_health_check(self):
        # A simple check to see if the app is alive, e.g. a 404 on root
        response = self.client.get("/")
        self.assertEqual(response.status_code, 404)

    def test_1_upload_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as tmp_file:
            tmp_file.write("hello world")
            tmp_file_path = tmp_file.name
            tmp_file_basename = os.path.basename(tmp_file_path)

        with open(tmp_file_path, "rb") as tf:
            response = self.client.post(
                "/upload", data={"file": (tf, tmp_file_basename)}
            )

        os.remove(tmp_file_path)  # Clean up the temp file from host

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("uploaded successfully", json_data.get("message", ""))

        # Verify file exists in the test_workspace
        uploaded_file_path = os.path.join(self.test_workspace.name, tmp_file_basename)
        self.assertTrue(os.path.exists(uploaded_file_path))
        with open(uploaded_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "hello world")

    def test_2_load_pickle_success(self):
        # Create a dummy pickle file in the test_workspace
        pickle_data = {"message": "hello from pickle"}
        pickle_filename = "test_data.pkl"
        pickle_filepath_in_workspace = os.path.join(
            self.test_workspace.name, pickle_filename
        )

        with open(pickle_filepath_in_workspace, "wb") as f:
            cloudpickle.dump(pickle_data, f)

        response = self.client.post("/load_pickle", json={"file_path": pickle_filename})
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("loaded successfully", json_data.get("message", ""))
        self.assertIn(pickle_filename, LOADED_PICKLES)
        self.assertEqual(LOADED_PICKLES[pickle_filename], pickle_data)

    def test_3_load_pickle_file_not_found(self):
        response = self.client.post(
            "/load_pickle", json={"file_path": "non_existent.pkl"}
        )
        self.assertEqual(response.status_code, 404)
        json_data = response.get_json()
        self.assertIn("Pickle file not found", json_data.get("error", ""))

    def test_4_execute_code_simple(self):
        code = "a = 1 + 2\nprint(a)"
        response = self.client.post("/execute", json={"code": code})
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data.get("stdout").strip(), "3")
        self.assertEqual(json_data.get("stderr"), "")

    def test_5_execute_code_with_loaded_pickle(self):
        # First, load a pickle
        pickle_data = {"value": 42}
        pickle_filename = "data_for_exec.pkl"
        pickle_filepath_in_workspace = os.path.join(
            self.test_workspace.name, pickle_filename
        )
        with open(pickle_filepath_in_workspace, "wb") as f:
            cloudpickle.dump(pickle_data, f)

        self.client.post("/load_pickle", json={"file_path": pickle_filename})  # Load it

        # Now, execute code that uses it
        code = (
            f"data = LOADED_PICKLES.get('{pickle_filename}')\n"
            "if data:\n"
            "    print(data['value'])\n"
            "else:\n"
            "    print('Pickle not found')"
        )
        response = self.client.post("/execute", json={"code": code})
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data.get("stdout").strip(), "42")

    def test_6_execute_code_accessing_uploaded_file(self):
        # 1. Upload a file
        text_content = "Hello from an uploaded file for execution!"
        text_filename = "exec_test_file.txt"

        # Create the file locally first
        temp_file_to_upload = os.path.join(self.temp_dir.name, text_filename)
        with open(temp_file_to_upload, "w") as f:
            f.write(text_content)

        # Upload it using the API
        with open(
            temp_file_to_upload, "rb"
        ) as f_rb:  # Flask test client needs rb for files
            upload_response = self.client.post(
                "/upload", data={"file": (f_rb, text_filename)}
            )
        self.assertEqual(upload_response.status_code, 200)

        # 2. Execute code that reads this file from the (monkeypatched) WORKSPACE_DIR
        # The path in the code should be relative to the sandbox's /workspace
        code_to_execute = (
            f"file_path = '/workspace/{text_filename}'\n"  # Code inside sandbox uses /workspace
            "try:\n"
            "    with open(file_path, 'r') as f:\n"
            "        content = f.read()\n"
            "    print(content)\n"
            "except FileNotFoundError:\n"
            "    print(f'File not found at {file_path}')"
        )

        # To make this test pass, the `sandbox_api.py`'s `execute_code` function
        # needs to be able to access files in `app.WORKSPACE_DIR` (which we've monkeypatched).
        # The `os.path.join(WORKSPACE_DIR, file.filename)` in `upload_file` uses the monkeypatched dir.
        # The code executed by `exec()` in `execute_code` runs in the context of `sandbox_api.py`.
        # If it tries to open `/workspace/exec_test_file.txt`, it will look for an actual `/workspace`
        # directory, *unless* the code itself is modified to use `app.WORKSPACE_DIR`.

        # The current `sandbox_api.py` `execute_code` doesn't alter file paths.
        # It executes code as is. So `open('/workspace/file.txt')` will try to open literally that.
        # This test will likely fail unless the test runner has write access to `/workspace`
        # or `sandbox_api.py` is changed to make file access relative to its `WORKSPACE_DIR` variable.

        # For now, let's assume the ideal scenario where the code execution environment
        # somehow respects this app.WORKSPACE_DIR. A more robust solution would be to
        # make the sandbox execute code within its WORKSPACE_DIR or provide a utility.
        # Given the current `sandbox_api.py`, let's adjust the code to reflect how it would work.
        # The code inside `exec` will run with `sandbox_api.py`'s current working directory.
        # We need to ensure the file path is correct relative to that.
        # The `upload_file` saves to `os.path.join(app.WORKSPACE_DIR, file.filename)`.
        # So the code should try to open `os.path.join(app.WORKSPACE_DIR, text_filename)`.
        # The simplest way is to have the code use the *actual* path that results from our monkeypatch.

        path_in_sandbox_script = os.path.join(app.WORKSPACE_DIR, text_filename)

        code_to_execute_corrected = (
            f"file_path = r'{path_in_sandbox_script}'\n"  # Use raw string for Windows paths if applicable
            "try:\n"
            "    with open(file_path, 'r') as f:\n"
            "        content = f.read()\n"
            "    print(content)\n"
            "except FileNotFoundError:\n"
            "    print(f'File not found at {file_path}')"
        )

        response = self.client.post(
            "/execute", json={"code": code_to_execute_corrected}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data.get("stdout").strip(), text_content)

    def test_7_execute_code_syntax_error(self):
        code = "print('hello world'\nthis is a syntax error"
        response = self.client.post("/execute", json={"code": code})
        self.assertEqual(response.status_code, 500)  # Server error due to exec failure
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("Error during execution", json_data.get("error"))
        self.assertNotEqual(
            json_data.get("stderr"), ""
        )  # Stderr should have exception info


if __name__ == "__main__":
    unittest.main()
