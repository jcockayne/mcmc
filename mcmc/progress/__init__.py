notebook = False
import jupyter_progress, stdout_progress

def get_default_progress():
    if notebook:
        try:
            import jupyter_progress
            return jupyter_progress.JupyterProgress()
        except Exception as ex:
            print('Unable to use JupyterProgress! An exception was raised:\n{}'.format(ex))
    import stdout_progress
    return stdout_progress.PrintProgress()
