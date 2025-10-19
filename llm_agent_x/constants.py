from os import getenv

from dotenv import load_dotenv

load_dotenv(".env", override=True)


LANGUAGE = "english"

redis_host = getenv("REDIS_HOST", None)
redis_port = int(getenv("REDIS_PORT", 6379)) # Keep default port, but host=None is the switch
redis_db = int(getenv("REDIS_DB", 0))
redis_expiry = int(getenv("REDIS_EXPIRY", 3600)) # Cache expiry in seconds (default 1 hour)

openai_base_url = getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
openai_api_key = getenv("OPENAI_API_KEY")
SANDBOX_API_URL = getenv("PYTHON_SANDBOX_API_URL", "http://localhost:5000")
