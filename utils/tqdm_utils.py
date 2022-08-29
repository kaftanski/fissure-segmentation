import contextlib
import inspect

import tqdm


@contextlib.contextmanager
def _redirect_to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.tqdm.write(*args, **kwargs)
        except:
            old_print(*args, ** kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def tqdm_redirect(*args, **kwargs):
    with _redirect_to_tqdm():
        for x in tqdm.tqdm(*args, **kwargs):
            yield x


if __name__ == '__main__':
    import time
    for i in tqdm_redirect(range(20)):
        time.sleep(.1)
        print(i)
