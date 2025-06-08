from os import getenv

from dotenv import load_dotenv

load_dotenv(".env", override=True)


LANGUAGE = "english"
redis_db = int(getenv("REDIS_DB", 0))  # Default to 0 if not set
redis_host = getenv("REDIS_HOST", "localhost")  # Default to localhost if not set
redis_expiry = int(
    getenv("REDIS_EXPIRY", 3600)
)  # Cache expiry in seconds (default 1 hour)
redis_port = int(getenv("REDIS_PORT", 6379))  # Default to 6379 if not set
openai_base_url = getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
openai_api_key = getenv("OPENAI_API_KEY")
SANDBOX_API_URL = getenv("PYTHON_SANDBOX_API_URL", "http://localhost:5000")
