""" Utilities for running scripts from an IDE as a background process with nohup. """

import os
import subprocess
import sys
from datetime import datetime

from utils.general_utils import new_dir

LOG_DIR = new_dir('./results', 'logs')


def maybe_run_detached_cli(args):
    if args.offline:
        # remove the --offline and the --show option (if given) from the arguments
        sys.argv.remove('--offline')
        try:
            sys.argv.remove('--show')
        except ValueError:
            pass

        # run script with modified argv
        _do_run(add_is_offline_arg=False)


def run_detached_from_pycharm():
    if '--is_offline' in sys.argv:
        # script has been called with nohup.
        # this prevents endless recursion of the scripts
        return
    else:
        _do_run(add_is_offline_arg=True)


def _do_run(add_is_offline_arg=True):
    script_name = os.path.split(sys.argv[0])[1]
    timestamp = datetime.now()
    readable_timestamp = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
    output_file = os.path.join(LOG_DIR, f'{script_name.replace(".py", "")}_{readable_timestamp}.txt')
    print(f'Running the script ({script_name}) in offline mode with nohup.\n'
          f'stdout will be redirected to {output_file}')

    for i, v in enumerate(sys.argv):
        if "--head_schedule" in v:
            sys.argv[i+1] = sys.argv[i+1].replace('"', '\\"')
            break

    # run the script in a subprocess with nohup
    subprocess.run(f'nohup {sys.executable} {" ".join(sys.argv)} {"--is_offline" if add_is_offline_arg else ""} >{output_file} &', shell=True)
    exit(0)
