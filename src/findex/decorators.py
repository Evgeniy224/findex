import time
import functools
import logging

# Налаштовуємо логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def timed(func):
    """
    Декоратор, який вимірює час виконання функції.
    Використовує functools.wraps, щоб зберегти ім'я та документацію оригінальної функції.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"Функція {func.__name__} виконана за {elapsed:.4f} сек")
        return result
    return wrapper