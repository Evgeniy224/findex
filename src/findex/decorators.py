import functools
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор, який логує час виконання функції у діагностичний логер."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug("Функція %s виконана за %.4f сек", func.__name__, elapsed)
        return result

    return wrapper
