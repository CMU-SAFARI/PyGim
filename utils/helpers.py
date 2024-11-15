__all__ = ['make_argument_parser', 'run_experiments', 'make_logger']

import argparse
import logging
import os
import shutil
import sys

import tqdm
import tqdm.contrib.logging


def make_argument_parser(result_name):
    root = argparse.ArgumentParser()

    root.add_argument("--log_file", type=str, default="./logs.log")
    root.add_argument("--append", action="store_true")
    root.add_argument("--verbose", action="store_true")
    root.add_argument("--dry_run", action="store_true")

    parser_group = root.add_subparsers(dest="action")

    parser = parser_group.add_parser("run")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--skip_failed", action="store_true")

    parser.add_argument("--src_root", type=str, default="./")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--build_root", type=str, default="./build")
    parser.add_argument("--result_root", type=str, default=f"./results/{result_name}")

    parser = parser_group.add_parser("migrate")

    parser.add_argument("--old", type=str, default=f"./results")
    parser.add_argument("--new", type=str, default=f"./results/{result_name}")

    parser = parser_group.add_parser("plot")
    parser.add_argument("--result_root", type=str, default=f"./results/{result_name}")
    root.add_argument("--figure_root", type=str, default="./")

    return root


def run_experiments(args, build_set, experiments, logger: logging.Logger, accept_failures: bool = False, repeat: int = 1):
    os.makedirs(args.result_root, exist_ok=True)
    # build_set = []
    # build_configs = set()
    # for experiment in experiments:
    #     config = (experiment.backend, experiment.dtype, experiment.sp_format)
    #     if config in build_configs:
    #         continue
    #     build_configs.add(config)
    #     build_set.append(experiment)
    #
    # build_set = generate_all_builds()
    for experiment in tqdm.tqdm(build_set,
                                leave=False,
                                dynamic_ncols=False,
                                ncols=0):
        experiment.build(src_root=args.src_root,
                         build_root=args.build_root,
                         force_rebuild=args.force_rebuild,
                         dry_run=args.dry_run,
                         logger=logger)
        

    for experiment in tqdm.tqdm(experiments,
                                leave=False,
                                dynamic_ncols=False,
                                ncols=0):
        if accept_failures:
            done_status = ('done', 'failed')
        else:
            done_status = ('done',)
        print(experiment)
        with tqdm.contrib.logging.logging_redirect_tqdm(loggers=[logger]):
            for handler in logger.handlers:
                tqdm_logger_class = getattr(tqdm.contrib.logging,
                                            '_TqdmLoggingHandler')
                if isinstance(handler, tqdm_logger_class):
                    handler.setLevel(
                        logging.DEBUG if args.verbose else logging.INFO)
            status = experiment.status_at(args.result_root)
            if status in done_status:
                if status == "failed":
                    logger.info(f"==> Skipping failed {experiment}")
                else:
                    logger.info(f"==> Skipping {experiment}")
                continue

            # experiment.build(src_root=args.src_root,
            #                  build_root=args.build_root,
            #                  force_rebuild=args.force_rebuild,
            #                  dry_run=args.dry_run,
            #                  logger=logger)
            experiment.run(src_root=args.src_root,
                           data_root=args.data_root,
                           build_root=args.build_root,
                           result_root=args.result_root,
                           repeat=repeat,
                           silent=args.skip_failed,
                           dry_run=args.dry_run,
                           logger=logger)


def migrate_experiments(args, experiments, logger: logging.Logger):
    os.makedirs(args.new, exist_ok=True)
    for experiment in tqdm.tqdm(experiments,
                                leave=False,
                                dynamic_ncols=False,
                                ncols=0):
        with tqdm.contrib.logging.logging_redirect_tqdm(loggers=[logger]):
            for handler in logger.handlers:
                tqdm_logger_class = getattr(tqdm.contrib.logging,
                                            '_TqdmLoggingHandler')
                if isinstance(handler, tqdm_logger_class):
                    handler.setLevel(
                        logging.DEBUG if args.verbose else logging.INFO)
            old_stdout_path = experiment.stdout_path_v1(args.old)
            old_stderr_path = experiment.stderr_path_v1(args.old)
            new_stdout_path = experiment.stdout_path(args.new)
            new_stderr_path = experiment.stderr_path(args.new)
            pairs = [
                (old_stdout_path, new_stdout_path),
                (old_stderr_path, new_stderr_path),
            ]

            for old_path, new_path in pairs:
                if os.path.exists(old_path):
                    logger.info(f"==> Migrating {old_path} to {new_path}")
                    if not args.dry_run:
                        shutil.move(old_path, new_path)
                elif os.path.exists(old_path + ".failed"):
                    logger.info(f"==> Migrating {old_path}.failed to {new_path}.failed")
                    if not args.dry_run:
                        shutil.move(old_path + ".failed", new_path + ".failed")
                else:
                    logger.info(f"==> Not found {old_path}")


def make_logger(name, args):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger.handlers[-1].setLevel(
        logging.DEBUG if args.verbose else logging.INFO)
    if args.log_file:
        logger.addHandler(
            logging.FileHandler(filename=args.log_file,
                                mode="a" if args.append else "w"))
        logger.handlers[-1].setLevel(logging.DEBUG)
    return logger
