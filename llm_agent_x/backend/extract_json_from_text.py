import json
def extract_json(s: str):
    """
    Extracts the first valid JSON object from a string,
    even if random characters are prepended or appended.
    """
    start = None
    brace_stack = 0

    for i, ch in enumerate(s):
        if ch == '{':
            if start is None:
                start = i
            brace_stack += 1
        elif ch == '}':
            brace_stack -= 1
            if start is not None and brace_stack == 0:
                candidate = s[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # keep searching, might not be complete yet
                    continue
    return None