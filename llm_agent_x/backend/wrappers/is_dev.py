from llm_agent_x.cli_args_parser import parser


def is_dev(func=None):
    def decorator(f):
        args = parser.parse_args()
        if args.dev_mode:
            return f

        def empty_func(*_args, **_kwargs):
            pass

        return empty_func

    # Allow both @is_dev and is_dev(func) styles
    return decorator(func) if func else decorator
