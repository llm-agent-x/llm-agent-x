# Installation

Follow these steps to install LLM Agent X and configure your environment.

## Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)
- Git (optional, for installing from source)

## Installation Steps

You can install LLM Agent X either from PyPI or from source.

### 1. Install the Package

**Option A: Install from PyPI (Recommended)**

This is the easiest way to get started:
```sh
pip install llm-agent-x
```

**Option B: Install from Source**

If you want to modify the code or contribute to development, install from source:

1.  Clone the repository:
    ```sh
    git clone https://github.com/cvaz1306/llm_agent_x.git
    cd llm_agent_x
    ```

2.  Install the package:
    *   For standard local development:
        ```sh
        pip install .
        ```
    *   To install in editable mode (changes to the source code will be immediately reflected):
        ```sh
        pip install -e .
        ```

### 2. Install `torch`

`torch` is a required runtime dependency but is not automatically installed by Poetry to allow for hardware-specific versions (e.g., CPU vs. GPU). You must install it manually.

Please visit the [PyTorch website](https://pytorch.org/get-started/locally/) to find the correct installation command for your operating system and hardware (CPU or specific CUDA version).

Example for a typical CPU-only installation:
```sh
pip install torch torchvision torchaudio
```

### 3. Set Up Environment Variables

LLM Agent X requires certain environment variables to be set for its operation. Create a `.env` file in the root directory of your project (if you installed from source) or in the directory where you will run `llm-agent-x` commands.

Add the following variables to your `.env` file, replacing placeholder values with your actual keys and preferences:

```env
# Required for web search functionality (if using SearXNG)
SEARX_HOST=http://localhost:8080

# Directory where output files will be saved
OUTPUT_DIR=./output/

# OpenAI API Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key

# Optional: Define a default LLM model to use if not specified by --model flag
# DEFAULT_LLM=gpt-4o-mini

# Optional: Configuration for the Python execution sandbox (if used)
# See docs/sandbox.md for more details.
# PYTHON_SANDBOX_API_URL=http://127.0.0.1:5000
```

**Explanation of Environment Variables:**

*   `SEARX_HOST`: The URL of your SearXNG instance. LLM Agent X uses this for the `web_search` tool via Brave Search integration, which can be configured to use a SearXNG instance.
*   `OUTPUT_DIR`: The default directory where generated reports and files will be saved.
*   `OPENAI_BASE_URL`: The base URL for the OpenAI API. Defaults to `https://api.openai.com/v1`.
*   `OPENAI_API_KEY`: Your API key for accessing OpenAI models.
*   `DEFAULT_LLM` (Optional): Specifies the default language model to be used (e.g., `gpt-4o-mini`, `gpt-4-turbo`). This can be overridden by the `--model` CLI argument.
*   `PYTHON_SANDBOX_API_URL` (Optional): The URL for the Python execution sandbox API. Only needed if you are using the sandbox feature.

After creating the `.env` file, ensure it is loaded by your environment. If you are running scripts directly, Python libraries like `python-dotenv` can load it automatically (LLM Agent X handles this internally when run as a CLI tool).

## Next Steps

With LLM Agent X installed and configured, you can now proceed to:

-   Learn about the [Command-Line Interface (CLI)](./cli.md).
-   Explore [API Usage](./api.md) for programmatic integration.
-   Set up the [Python Sandbox](./sandbox.md) if you need isolated code execution.
