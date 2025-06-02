def exec_python(code, globals=None, locals=None):
    """
    Execute the given code with empty globals and locals.

    This function executes the provided code string in an isolated
    environment where both the global and local namespaces are empty,
    preventing the code from accessing or modifying external variables.

    Parameters:
    code (str): The code to be executed.
    globals (dict, optional): A dictionary of global variables. Defaults to None.
    locals (dict, optional): A dictionary of local variables. Defaults to None.

    Returns:
    None
    """
    exec(code, {}, {})