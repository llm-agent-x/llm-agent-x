import uuid


def int_to_base26(num):
    if num < 0:
        raise ValueError("Number must be non-negative")

    result = []
    while num >= 0:
        num, remainder = divmod(num, 26)
        result.append(chr(remainder + 65))  # Convert 0-25 to 'A'-'Z'
        if num == 0:
            break
        num -= 1  # Adjust for base-26 indexing

    return "".join(reversed(result))


def batch_random_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]
