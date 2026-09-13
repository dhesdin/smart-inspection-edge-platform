import datetime


def get_run_id(tag: str | None = None) -> str:
    """Generate a unique run ID based on the current date and time, optionally prefixed with a tag.

    Args:
        tag (str | None): Optional tag to prefix the run ID with.

    Returns:
        str: The generated run ID.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"{tag}_{now}"
    return now
