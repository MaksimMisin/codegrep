import logging


def get_logger():
    """Get the codegrep logger with consistent configuration."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger("codegrep")
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
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    if not logger.handlers:  # Only add handler if none exist
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
