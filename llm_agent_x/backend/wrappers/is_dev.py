import functools
from llm_agent_x.cli_args_parser import parser

def is_dev(func=None):
    def decorator(f):
        # Use functools.wraps to preserve the original function's metadata
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Parse arguments ONLY when the decorated function is called
            # We use parse_known_args to avoid errors if other args are present
            cli_args, _ = parser.parse_known_args()
            if cli_args.dev_mode:
                # If in dev mode, execute the original function (e.g., `ic`)
                return f(*args, **kwargs)
            else:
                # If not in dev mode, do nothing.
                pass
        return wrapper

    # This logic allows the decorator to be used with or without parentheses: @is_dev or @is_dev()
    if func:
        return decorator(func)
    return decorator