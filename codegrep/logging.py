import logging


def get_logger():
    """Get the codegrep logger with consistent configuration."""
    logger = logging.getLogger("codegrep")

    # Only configure if the logger hasn't been set up yet
    if not logger.handlers:
        # Configure the logger
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Silence third-party library logging
        for logger_name in [
            "httpx",
            "openai",
            "httpcore",
            "httpcore._backends.sync",
            "langchain_core.callbacks.manager",
            "faiss",
            "urllib3",
            "requests",
        ]:
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.setLevel(logging.WARNING)
            # Remove any existing handlers
            third_party_logger.handlers = []
            third_party_logger.propagate = False

    return logger
