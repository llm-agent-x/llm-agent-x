# Python Execution Sandbox

This is a Dockerized sandbox environment for executing Python code. This provides isolation and allows for pre-loading files and cloudpickle objects into the execution namespace.

## Building the Sandbox Image
From the root of the repository:
```sh
docker build -t python-sandbox ./sandbox
```

## Running the Sandbox Image

You can run the container without requiring auth, or require an API token.

Without authentication:
```sh
docker run -d -p 5000:5000 --rm python-sandbox
```

With authentication:
```sh
docker run -d -p 5000:5000 --rm \
    -e PYTHON_SANDBOX_API_TOKEN_REQUIRED=true \
    -e PYTHON_SANDBOX_API_TOKEN=your_token_here \
    python-sandbox
```

Or to generate a random token:
```sh
token=$(openssl rand -hex 32)
echo "Your token: $token"
docker run -d -p 5000:5000 --rm \
    -e PYTHON_SANDBOX_API_TOKEN_REQUIRED=true \
    -e PYTHON_SANDBOX_API_TOKEN="$token" \
    python-sandbox
```