from collections.abc import Iterable
from pathlib import Path
import re





def ensure_iterable(data):
    """Ensures the input is returned as an iterable (list) if it is not already iterable.

    Args:
        data (Any): The input value to be checked.

    Returns:
        Iterable: The input wrapped in a list if necessary, or returned unchanged if already iterable.
    """

    if isinstance(data, (str, Path)) or not isinstance(data, Iterable):
        return [data]
    return data





def get_title_and_units(s):
    """Extracts title and units from an input of the format Title (units).

    Args:
        s (str): A string of the form Title (units).

    Returns:
        tuple[str, str | None]: A tuple where the first element is the title (part outside the parenthesis), and the second string is the units (part within the parenthesis), or None if there are no parenthesis.
    """

    # Regular expression to capture the part before the parentheses and the part inside them
    pattern = r'(.*?)\s?\((.*?)\)'
    
    # Search for the pattern in the input string
    match = re.search(pattern, s)
    
    if match:
        # Extract the words outside and the words inside the parentheses
        outside = match.group(1).strip()
        inside = match.group(2).strip()
        return outside, inside
    else:
        return s, None  # If no parentheses are found