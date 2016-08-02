import jupyter_progress, stdout_progress, null_progress, base

notebook = False
verbose = True


def factory(obj):
    if obj is None:
        return get_default_progress()
    if isinstance(obj, base.ProgressBase):
        return obj
    if obj.lower() == 'none':
        return null_progress.NullProgress()
    if obj.lower == 'jupyter':
        return jupyter_progress.JupyterProgress(verbosity=verbose)
    if obj.lower == 'stdout' or obj.lower == 'console' or obj.lower == 'print':
        return stdout_progress.PrintProgress(verbosity=verbose)
    raise Exception("Don't understand object {}".format(obj))


def get_default_progress():
    if notebook:
        try:
            import jupyter_progress
            return jupyter_progress.JupyterProgress(verbosity=verbose)
        except Exception as ex:
            print('Unable to use JupyterProgress! An exception was raised:\n{}'.format(ex))
    import stdout_progress
    return stdout_progress.PrintProgress()

