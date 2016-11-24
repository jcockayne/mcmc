import jupyter_progress, stdout_progress, null_progress, base

verbose = True
PROGRESS_NOTEBOOK = 'notebook'
PROGRESS_STDOUT = 'stdout'
PROGRESS_NONE = 'none'

default_progress = PROGRESS_STDOUT


def factory(obj):
    if obj is None:
        return get_default_progress()
    if isinstance(obj, base.ProgressBase):
        return obj
    if obj.lower() == PROGRESS_NONE:
        return null_progress.NullProgress()
    if obj.lower() == PROGRESS_NOTEBOOK:
        return jupyter_progress.JupyterProgress(verbosity=verbose)
    if obj.lower() == PROGRESS_STDOUT or obj.lower == 'console' or obj.lower == 'print':
        return stdout_progress.PrintProgress(verbosity=verbose)
    raise Exception("Don't understand object {}".format(obj))


def get_default_progress():
    return factory(default_progress)
