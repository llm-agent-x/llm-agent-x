import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure the path is correct to import exec_python
# Assuming this test file is in llm_agent_x/tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import requests  # Moved import to top
from llm_agent_x.tools.exec_python import exec_python


# Define a helper function to simulate sandbox API responses
def mock_sandbox_response(status_code, json_data):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json = MagicMock(return_value=json_data)
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
    return mock_resp


class TestExecPythonSandbox(unittest.TestCase):

    @patch("llm_agent_x.tools.exec_python.requests.post")
    def test_1_simple_code_execution_sandbox(self, mock_post):
        # Mock the /execute endpoint
        mock_post.return_value = mock_sandbox_response(
            200, {"stdout": "Hello from sandbox!", "stderr": "", "error": None}
        )

        code = "print('Hello from sandbox!')"
        result = exec_python(code, use_docker_sandbox=True)

        mock_post.assert_called_once_with(
            "http://localhost:5000/execute", json={"code": code}
        )
        self.assertEqual(result["stdout"], "Hello from sandbox!")
        self.assertEqual(result["error"], None)

    @patch("llm_agent_x.tools.exec_python.requests.post")
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data="file content"
    )
    @patch("os.path.basename", return_value="testfile.txt")
    def test_2_upload_file_and_execute(self, mock_basename, mock_open_file, mock_post):
        # Simulate a sequence of calls: /upload then /execute
        mock_upload_response = mock_sandbox_response(200, {"message": "File uploaded"})
        mock_execute_response = mock_sandbox_response(
            200, {"stdout": "File content: file content", "stderr": "", "error": None}
        )
        mock_post.side_effect = [mock_upload_response, mock_execute_response]

        file_to_upload = "dummy_path/testfile.txt"
        code = "with open('/workspace/testfile.txt', 'r') as f: print(f.read())"

        result = exec_python(
            code, use_docker_sandbox=True, files_to_upload=[file_to_upload]
        )

        self.assertEqual(mock_post.call_count, 2)
        mock_post.assert_any_call(
            "http://localhost:5000/upload",
            files={"file": (mock_basename.return_value, mock_open_file.return_value)},
        )
        mock_post.assert_any_call("http://localhost:5000/execute", json={"code": code})
        self.assertEqual(result["stdout"], "File content: file content")
        self.assertEqual(result["error"], None)
        mock_open_file.assert_called_once_with(file_to_upload, "rb")

    @patch("llm_agent_x.tools.exec_python.requests.post")
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data="pickle data"
    )
    @patch("os.path.basename", return_value="my_data.pkl")
    def test_3_upload_and_load_pickle_and_execute(
        self, mock_basename, mock_open_file, mock_post
    ):
        # Simulate /upload, then /load_pickle, then /execute
        mock_upload_response = mock_sandbox_response(
            200, {"message": "Pickle uploaded"}
        )
        mock_load_pickle_response = mock_sandbox_response(
            200, {"message": "Pickle loaded"}
        )
        mock_execute_response = mock_sandbox_response(
            200, {"stdout": "Pickle value: test", "stderr": "", "error": None}
        )
        mock_post.side_effect = [
            mock_upload_response,
            mock_load_pickle_response,
            mock_execute_response,
        ]

        pickle_file_local_path = "dummy_path/my_data.pkl"
        pickle_file_sandbox_path = "my_data.pkl"  # Relative path in sandbox
        code = "print(LOADED_PICKLES['my_data.pkl'])"

        result = exec_python(
            code,
            use_docker_sandbox=True,
            files_to_upload=[pickle_file_local_path],
            cloud_pickle_files_to_load=[pickle_file_sandbox_path],
        )

        self.assertEqual(mock_post.call_count, 3)
        # Check upload call
        mock_post.assert_any_call(
            "http://localhost:5000/upload",
            files={"file": (mock_basename.return_value, mock_open_file.return_value)},
        )
        # Check load_pickle call
        mock_post.assert_any_call(
            "http://localhost:5000/load_pickle",
            json={"file_path": pickle_file_sandbox_path},
        )
        # Check execute call
        mock_post.assert_any_call("http://localhost:5000/execute", json={"code": code})
        self.assertEqual(result["stdout"], "Pickle value: test")
        self.assertEqual(result["error"], None)
        mock_open_file.assert_called_once_with(pickle_file_local_path, "rb")

    @patch("llm_agent_x.tools.exec_python.requests.post")
    @patch("builtins.open", side_effect=FileNotFoundError("Mocked FileNotFoundError"))
    def test_4_sandbox_api_upload_file_not_found(self, mock_open, mock_post):
        # Simulate FileNotFoundError when trying to open a file for upload
        result = exec_python(
            "print('test')",
            use_docker_sandbox=True,
            files_to_upload=["non_existent_file.txt"],
        )

        mock_open.assert_called_once_with("non_existent_file.txt", "rb")
        mock_post.assert_not_called()  # requests.post should not be called if file open fails
        self.assertTrue(
            "Error: File not found for upload: non_existent_file.txt"
            in result["stderr"]
        )
        self.assertEqual(result["error"], "File upload error")

    @patch("llm_agent_x.tools.exec_python.requests.post")
    def test_5_sandbox_api_pickle_load_error(self, mock_post):
        # Simulate successful upload, but error on load_pickle
        mock_upload_response = mock_sandbox_response(200, {"message": "File uploaded"})

        # Mock for the error response from /load_pickle
        error_response_mock = MagicMock()
        error_response_mock.status_code = 500
        error_response_mock.json.return_value = {
            "error": "Pickle decode error",
            "trace": "some traceback",
        }
        error_response_mock.raise_for_status.side_effect = (
            requests.exceptions.HTTPError(response=error_response_mock)
        )

        mock_post.side_effect = [mock_upload_response, error_response_mock]

        # Patch open to avoid actual file operations for this test's focus
        with patch(
            "builtins.open", new_callable=unittest.mock.mock_open, read_data="data"
        ):
            result = exec_python(
                "print('test')",
                use_docker_sandbox=True,
                files_to_upload=["dummy.pkl"],
                cloud_pickle_files_to_load=["dummy.pkl"],
            )

        self.assertTrue("Error loading cloudpickle file dummy.pkl" in result["stderr"])
        self.assertTrue(
            "Pickle decode error" in result["stderr"]
        )  # Check for sandbox error message
        self.assertEqual(result["error"], "Cloudpickle load error")

    @patch("llm_agent_x.tools.exec_python.requests.post")
    def test_6_sandbox_api_execute_error(self, mock_post):
        import requests

        # Simulate error on /execute
        error_response_mock = MagicMock()
        error_response_mock.status_code = 500
        error_response_mock.json.return_value = {
            "stdout": "",
            "stderr": "Syntax Error!",
            "error": "Execution failed",
            "trace": "detail trace",
        }
        error_response_mock.raise_for_status.side_effect = (
            requests.exceptions.HTTPError(response=error_response_mock)
        )

        mock_post.return_value = error_response_mock

        result = exec_python("print(1/0)", use_docker_sandbox=True)

        self.assertTrue("Syntax Error!" in result["stderr"])
        self.assertTrue(
            "Execution error from sandbox: Execution failed" in result["stderr"]
        )
        self.assertEqual(result["error"], "Execution failed")

    def test_7_local_execution_simple(self):
        # Test the original local execution path (no mocking needed for requests)
        code = "x = 10; y = 20; z = x + y"  # No print, just exec
        # For local exec, we expect placeholder stdout/stderr as per current implementation
        result = exec_python(code, use_docker_sandbox=False)

        self.assertEqual(result["stdout"], "[Local execution - stdout not captured]")
        self.assertEqual(result["stderr"], "[Local execution - stderr not captured]")
        self.assertIsNone(result["error"])

    def test_8_local_execution_with_error(self):
        code = "1/0"  # This will raise ZeroDivisionError
        result = exec_python(code, use_docker_sandbox=False)

        self.assertEqual(result["stdout"], "")
        self.assertTrue("division by zero" in result["stderr"])
        self.assertTrue("division by zero" in result["error"])

    @patch("llm_agent_x.tools.exec_python.os.getenv", return_value=None)
    @patch("llm_agent_x.tools.exec_python.requests.post")
    def test_9_sandbox_url_not_configured(self, mock_requests_post, mock_getenv):
        code = "print('Hello from sandbox!')"
        result = exec_python(code, use_docker_sandbox=True)

        mock_requests_post.assert_not_called()  # Requests should not be called if URL is missing
        self.assertEqual(
            result["stderr"], "PYTHON_SANDBOX_API_URL environment variable is not set."
        )
        self.assertEqual(result["error"], "Configuration error")

    # Example of testing actual request failure after file is successfully opened
    @patch("llm_agent_x.tools.exec_python.requests.post")
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data="file content"
    )
    @patch("os.path.basename", return_value="testfile.txt")
    def test_10_sandbox_api_upload_request_fails(
        self, mock_basename, mock_open, mock_post
    ):
        # Simulate an error during the requests.post call for upload
        mock_post.side_effect = requests.exceptions.RequestException(
            "Simulated network error during upload"
        )

        result = exec_python(
            "print('test')",
            use_docker_sandbox=True,
            files_to_upload=[
                "existent_file.txt"
            ],  # File assumed to be successfully opened by mock_open
        )

        mock_open.assert_called_once_with("existent_file.txt", "rb")
        mock_post.assert_called_once()  # Ensure requests.post was attempted
        self.assertTrue(
            "Error uploading existent_file.txt: Simulated network error during upload"
            in result["stderr"]
        )
        self.assertEqual(result["error"], "File upload error")


if __name__ == "__main__":
    unittest.main()
