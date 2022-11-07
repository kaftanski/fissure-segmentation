import json
import os

from argparse import Namespace


def store_args(args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'commandline_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args_dict(from_dir):
    args_file = os.path.join(from_dir, 'commandline_args.json')
    if not os.path.isfile(args_file):
        return None

    with open(args_file, 'r') as f:
        args_from_file = json.load(f)

    return args_from_file


def load_args_for_testing(from_dir, current_args):
    args_from_file = load_args_dict(from_dir)
    if args_from_file is None:
        # compatibility for older training runs, where no file has been created
        store_args(current_args, from_dir)
        return current_args

    # set the arguments that should be overwritten from the test call
    args_from_file['test_only'] = current_args.test_only
    args_from_file['show'] = current_args.show
    args_from_file['gpu'] = current_args.gpu
    args_from_file['fold'] = current_args.fold

    # add keys that may have been added since the training run
    for key in current_args.__dict__.keys():
        if key not in args_from_file.keys():
            args_from_file[key] = getattr(current_args, key)

    return Namespace(**args_from_file)
