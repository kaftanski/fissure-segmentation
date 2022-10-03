import os
import subprocess
import sys
from datetime import datetime

from utils.utils import new_dir

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
        run_detached_from_pycharm()


def run_detached_from_pycharm():
    script_name = os.path.split(sys.argv[0])[1]
    timestamp = datetime.now()
    readable_timestamp = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
    output_file = os.path.join(LOG_DIR, f'{script_name.replace(".py", "")}_{readable_timestamp}.txt')
    print(f'Running the script ({script_name}) in offline mode with nohup.\n'
          f'stdout will be redirected to {output_file}')

    # run the script in a subprocess with nohup
    subprocess.run(f'nohup {sys.executable} {" ".join(sys.argv)} >{output_file} &', shell=True)
    exit(0)
