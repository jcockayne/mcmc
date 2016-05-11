import progress
from pcn import pCN
from rwm import rwm
from hmc import hmc

def load_ipython_extension(shell):
    # The `ipython` argument is the currently active `InteractiveShell`
    # instance, which can be used in any way. This allows you to register
    # new magics or aliases, for example.
    shell.register_magic_function(mcmc)


def mcmc(line):
    if line == 'notebook':
        progress.notebook = True
