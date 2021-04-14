import sys
from functools import wraps


def print_banner(s, width=80, banner_token='-'):
    if len(s) > width:
        return s
    rem = width-len(s)
    rhs = rem//2
    lhs = rem-rhs
    if rhs > 0:
        rhs_pad = " " + (rhs-1)*banner_token
    else:
        rhs_pad = ""
    lhs_pad = (lhs-1)*banner_token + " "
    print(lhs_pad + s + rhs_pad, file=sys.stderr)


def print_banner_completion_wrapper(s, width=80, banner_token='-'):
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print_banner(s, width, banner_token)
            result = func(*args, **kwargs)
            print_banner("Done " + s, width, banner_token)
            print(file=sys.stderr)
            return result
        return wrapper
    return wrap


def print_error(*args):
    print(*args, file=sys.stderr)
