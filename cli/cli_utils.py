import json
import os
from argparse import ArgumentParser


def store_args(args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'commandline_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args_for_testing(from_dir, current_args):
    args_file = os.path.join(from_dir, 'commandline_args.json')
    if not os.path.isfile(args_file):
        # compatibility for older training runs, where no file has been created
        store_args(current_args, from_dir)
        return current_args

    parser = ArgumentParser()
    args_from_file = parser.parse_args()
    with open(args_file, 'r') as f:
        args_from_file.__dict__ = json.load(f)

    # set the arguments that should be overwritten from the test call
    args_from_file.test_only = current_args.test_only
    args_from_file.show = current_args.show
    args_from_file.gpu = current_args.gpu
    args_from_file.fold = current_args.fold

    return args_from_file
