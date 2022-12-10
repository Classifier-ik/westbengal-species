from functools import wraps
from flask import redirect, session, url_for


def login_required(category):
    """
    Check is completely dependent on session variables.
    First check if 'logged_in' is True or not.
    Then check  if the category is matching or not.
    category can be a list (for different kind of users),
    or a str (for single kind of user).
    """
    def actual_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not session.get('logged_in'):
                return redirect(url_for('index'))
            if isinstance(category, list):
                if not session.get('category') in category:
                    return redirect(url_for('index'))
                else:
                    return fn(*args, **kwargs)
            elif isinstance(category, str):
                if not session.get('category') == category:
                    return redirect(url_for('index'))
                else:
                    return fn(*args, **kwargs)
            else:
                # type of category not known. Can't proceed.
                return redirect(url_for('index'))
        return wrapper
    return actual_decorator