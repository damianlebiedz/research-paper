import logging
import sys


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(module)s.%(funcName)s(): %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
