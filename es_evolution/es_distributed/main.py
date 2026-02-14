import errno
import json
import logging
import os
import sys

import click

from .dist import RelayClient
from .es import run_master, run_worker, SharedNoiseTable


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@click.group()
def cli():
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stderr)


@cli.command()
@click.option('--exp_str')
@click.option('--exp_file')
@click.option('--master_host', default='localhost')  # Added this
@click.option('--master_port', default=6379, type=int)  # Added this
@click.option('--master_socket_path')  # Made this optional
@click.option('--log_dir')
def master(exp_str, exp_file, master_host, master_port, master_socket_path, log_dir):
    import datetime
    # Start the master
    assert (exp_str is None) != (exp_file is None), 'Must provide exp_str xor exp_file to the master'
    if exp_str:
        exp = json.loads(exp_str)
    elif exp_file:
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
    else:
        assert False

    # --- AUTO-TIMESTAMP LOGIC ---
    # Create a unique name like "Cheetah_20251229_1305"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_prefix = exp.get('exp_prefix', 'exp')
    unique_name = f"{exp_prefix}_{timestamp}"

    # If no log_dir provided, use our local 'logs' folder with the unique name
    if not log_dir:
        log_dir = os.path.join('logs', unique_name)
    else:
        log_dir = os.path.join(os.path.expanduser(log_dir), unique_name)

    mkdir_p(log_dir)
    logging.info(f'Logging results to: {log_dir}')
    # ----------------------------
    # --- NEW: SAVE CONFIG FOR PERMANENT RECORD ---
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(exp, f, indent=4)
    # ---------------------------------------------

    # Decide whether to use Socket or Host/Port
    if master_socket_path:
        redis_cfg = {'unix_socket_path': master_socket_path}
    else:
        redis_cfg = {'host': master_host, 'port': master_port}

    run_master(redis_cfg, log_dir, exp)


@cli.command()
@click.option('--master_host', default='localhost')
@click.option('--master_port', default=6379, type=int)
@click.option('--relay_socket_path')  # Remove required=True
@click.option('--num_workers', type=int, default=0)
@click.option('--seed', type=int, default=1)  # 1. This tells Click to look for --seed
def workers(master_host, master_port, relay_socket_path, num_workers, seed):
    # We are bypassing the RelayClient for local testing
    master_redis_cfg = {'host': master_host, 'port': master_port}

    # Start the workers directly connecting to Master Redis
    # noise = SharedNoiseTable()
    noise = SharedNoiseTable(seed=seed)
    num_workers = num_workers if num_workers else os.cpu_count()
    logging.info('Spawning {} workers connecting directly to {}:{}'.format(num_workers, master_host, master_port))

    for _ in range(num_workers):
        if os.fork() == 0:
            # We pass master_redis_cfg instead of relay_cfg
            run_worker(master_redis_cfg, noise=noise)
            return
    os.wait()


if __name__ == '__main__':
    cli()
