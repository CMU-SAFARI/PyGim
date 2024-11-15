__all__ = ['Experiment']

import collections
import dataclasses
import logging
import os.path
import re
import shutil
from typing import Optional
from .autotuner import autotune
import numpy as np

BLNC_CSR_DEF = {
    "nnz": "-DBLNC_ROW=0 -DBLNC_NNZ_RGRN=0 -DBLNC_NNZ=1",
    "row": "-DBLNC_ROW=1 -DBLNC_NNZ_RGRN=0 -DBLNC_NNZ=0",
}
BLNC_COO_DEF = {
    "nnz": "-DBLNC_ROW=0  -DBLNC_NNZ_RGRN=0  -DBLNC_NNZ=1",
    "row": "-DBLNC_ROW=0  -DBLNC_NNZ_RGRN=1  -DBLNC_NNZ=0",
}
BLNC_TSKLT_CSR_DEF = {
    "nnz": "-DBLNC_TSKLT_ROW=0 -DBLNC_TSKLT_NNZ=1",
    "row": "-DBLNC_TSKLT_ROW=1 -DBLNC_TSKLT_NNZ=0",
}
BLNC_TSKLT_COO_DEF = {
    "nnz": "-DBLNC_TSKLT_ROW=0 -DBLNC_TSKLT_NNZ_RGRN=0 -DBLNC_TSKLT_NNZ=1",
    "row": "-DBLNC_TSKLT_ROW=0 -DBLNC_TSKLT_NNZ_RGRN=1 -DBLNC_TSKLT_NNZ=0",
}
CG_LOCK_DEF = {
    True: "-DLOCKFREEV2=0 -DCG_LOCK=1",
    False: "-DLOCKFREEV2=1 -DCG_LOCK=0",
}
MERGE_DEF = {
    "block": "-DROW_MERGE=0 -DBLOCK_MERGE=1",
    "row": "-DROW_MERGE=1 -DBLOCK_MERGE=0",
}
SYNC_DEF = {
    "async": "-DSYNC=0",
    "sync": "-DSYNC=1",
}
TYPES_BYTES = {"INT64": 8, "INT32": 4, "INT16": 2, "INT8": 1, "FLT32": 4, "DBL64": 8}


def run(cmd,
        workdir: str = ".",
        decode: bool = True,
        shell: bool = False,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
        silent: bool = False,
        dry_run: bool = True,
        logger: Optional[logging.Logger] = None):
    if isinstance(cmd, str):
        cmd = [cmd]
    import subprocess
    if logger is not None:
        logger.debug(f"Running: {cmd}")
    stdout = None
    stderr = None
    if not dry_run:
        pipe = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if capture_stdout else None,
            stderr=subprocess.PIPE if capture_stderr else None,
            cwd=workdir,
            shell=shell)
        returncode = pipe.returncode
        if capture_stdout:
            stdout = pipe.stdout
        if capture_stderr:
            stderr = pipe.stderr
    else:
        returncode = 0
        if capture_stdout:
            stdout = b'<dummy stdout>'
        if capture_stderr:
            stderr = b'<dummy stderr>'

    if returncode:
        message = f"The following command failed with return code {returncode}\n" \
                  f"==> {' '.join(cmd)}\n" \
                  f"==> cwd: {os.path.abspath(workdir)}\n"
        if stdout is not None:
            message += f"\n==> stdout:\n{stdout.decode()}"
        if stderr is not None:
            message += f"\n==> stderr:\n{stderr.decode()}"
        message += "\n"
        if silent:
            if logger is not None:
                logger.critical(message)
        else:
            raise RuntimeError(message)
    if decode:
        if stdout is not None and capture_stdout:
            stdout = stdout.decode()
        if stdout is not None and capture_stderr:
            stderr = stderr.decode()
    return {'stdout': stdout, 'stderr': stderr, 'return': returncode}


def build_lib(src_path: str,
              build_path: str,
              dtype: str,
              balance: str,
              cache_size: int,
              sp_format: str,
              balance_tsklt: str,
              nr_tasklets: int,
              cg_lock: bool,
              merge: str,
              sync: bool,
              op: str = "MUL",
              dry_run: bool = False,
              logger: Optional[logging.Logger] = None):

    if not dry_run:
        os.makedirs(build_path, exist_ok=True)
    if sp_format == "COO":
        blnc_def = BLNC_COO_DEF[balance]
        blnc_tasklt_def = BLNC_TSKLT_COO_DEF[balance_tsklt]
    else:
        blnc_def = BLNC_CSR_DEF[balance]
        blnc_tasklt_def = BLNC_TSKLT_CSR_DEF[balance_tsklt]
    cmake_def = f"-D NR_TASKLETS={nr_tasklets} " \
                f"-D KERNEL=\"{op}_{sp_format}\" " \
                f"-D extra_def=\"-D{dtype}=1 {blnc_def} {blnc_tasklt_def} {CG_LOCK_DEF[cg_lock]} {MERGE_DEF[merge]} " \
                f"{SYNC_DEF['sync' if sync else 'async']} -DPIM_SEQREAD_CACHE_SIZE={cache_size}\" "
    # format(NR_TASKLET, op, sp_format, dtype, blnc_def, blnc_tasklt_def, CG_LOCK_DEF[cg_lock])
    src_path = os.path.abspath(src_path)
    run("cmake {} {}".format(cmake_def, src_path),
        shell=True,
        capture_stderr=True,
        capture_stdout=True,
        logger=logger,
        dry_run=dry_run,
        workdir=build_path)
    run("make -j",
        shell=True,
        capture_stdout=True,
        capture_stderr=True,
        logger=logger,
        dry_run=dry_run,
        workdir=build_path)


@dataclasses.dataclass
class Experiment(object):
    dataset: str
    sp_part: int
    ds_part: int

    sp_format: str
    dense_size: int
    dtype: str
    balance: str
    balance_tsklt: str
    nr_tasklets: int
    cg_lock: bool
    cache_size: int

    backend: Optional[str] = None

    merge: str = "block"
    groups_per_rank: Optional[int] = None
    sync: bool = True
    tune: str = 'FALSE'
    nr_dpus: Optional[int] = None

    # This two parameters are for experiment 5
    model: Optional[str] = None
    num_layers: Optional[int] = None

    def __post_init__(self):
        if self.backend is None:
            self.backend = self.default_backend

    @property
    def build_params(self):
        build_keys = [
            'backend',
            'sp_format',
            'dtype',
            'balance',
            'balance_tsklt',
            'nr_tasklets',
            'cg_lock',
            'cache_size',
            'merge',
            'sync',
        ]
        return {k: getattr(self, k) for k in build_keys}

    # @property
    # def frozen_build_params_v1(self):
    #     params = list()
    #     params.extend([
    #         ('spf', f"{self.sp_format}"),
    #         ('dtype', f"{self.dtype}"),
    #         ('blnc', f"{self.balance}"),
    #         ('blnc_tsklt', f"{self.balance_tsklt}"),
    #         ('nr_tasklets', f"{self.nr_tasklets}"),
    #         ('cg_lock', f"{self.cg_lock}"),
    #         ('cache_size', f"{self.cache_size}"),
    #     ])
    #     if self.groups_per_rank is not None and self.groups_per_rank != 1:
    #         params.append(('gpr', f"{self.groups_per_rank}"))
    #     if self.merge != "block":
    #         params.append(('merge', f"{self.merge}"))
    #     if not self.sync:
    #         params.append(('sync', f"{self.sync}"))
    #     return collections.OrderedDict(params)

    @property
    def frozen_build_params(self):
        params = list()
        # if self.backend != self.default_backend:
        #     params.append(('backend', f"{self.backend}"))
        params.extend([
            ('backend', f"{self.backend}"),
            ('spf', f"{self.sp_format}"),
            ('dtype', f"{self.dtype}"),
            ('blnc', f"{self.balance}"),
            ('blnc_tsklt', f"{self.balance_tsklt}"),
            ('nr_tasklets', f"{self.nr_tasklets}"),
            ('cg_lock', f"{self.cg_lock}"),
            ('cache_size', f"{self.cache_size}"),
        ])
        if self.groups_per_rank is not None and self.groups_per_rank != 1:
            params.append(('gpr', f"{self.groups_per_rank}"))
        if self.merge != "block":
            params.append(('merge', f"{self.merge}"))
        if not self.sync:
            params.append(('sync', f"{self.sync}"))
        return collections.OrderedDict(params)

    # @property
    # def frozen_run_params_v1(self):
    #     params = self.frozen_build_params_v1
    #     cache_size = params.pop('cache_size')
    #     params = list(params.items())
    #     params.append(('spds', f"{self.sp_part}x{self.ds_part}"))
    #     params.append(('dataset', f"{self.dataset}"))
    #     params.append(('dense_size', f"{self.dense_size}"))
    #     params.append(('nr_dpus', f"{self.nr_dpus}"))
    #     params.append(('cache_size', cache_size))
    #     if self.model is not None and self.num_layers is not None:
    #         raise RuntimeError("frozen_build_params_v1 does not accept model and num_layers parameter")
    #     return collections.OrderedDict(params)

    @property
    def frozen_run_params(self):
        params = self.frozen_build_params
        cache_size = params.pop('cache_size')
        params = list(params.items())
        params.append(('spds', f"{self.sp_part}x{self.ds_part}"))
        params.append(('dataset', f"{self.dataset}"))
        params.append(('dense_size', f"{self.dense_size}"))
        params.append(('nr_dpus', f"{self.nr_dpus}"))
        params.append(('cache_size', cache_size))
        if self.model is not None:
            params.append(('model', f"{self.model}"))
        if self.num_layers is not None:
            params.append(('num_layers', f"{self.num_layers}"))
        return collections.OrderedDict(params)

    @property
    def default_backend(self):
        if self.groups_per_rank is not None and self.groups_per_rank != 1:
            return "backend_pim_multigroup"
        if self.ds_part == 0:
            return "backend_pim_grande"
        return "backend_pim_group"

    def build_path(self, build_root: str):
        build_params = '-'.join(
            [f'{k}={v}' for k, v in self.frozen_build_params.items()])
        return os.path.join(build_root, build_params)

    def stdout_path(self, result_root: str, failed: bool = False):
        run_params = '-'.join(
            [f'{k}={v}' for k, v in self.frozen_run_params.items()])
        path = os.path.join(result_root, run_params + '.out')
        if failed:
            path += ".failed"
        return path

    # def stdout_path_v1(self, result_root: str, failed: bool = False):
    #     run_params = '-'.join(
    #         [f'{k}={v}' for k, v in self.frozen_run_params_v1.items()])
    #     path = os.path.join(result_root, run_params + '.out')
    #     if failed:
    #         path += ".failed"
    #     return path

    def stderr_path(self, result_root: str, failed: bool = False):
        run_params = '-'.join(
            [f'{k}={v}' for k, v in self.frozen_run_params.items()])
        path = os.path.join(result_root, run_params + '.err')
        if failed:
            path += ".failed"
        return path

    # def stderr_path_v1(self, result_root: str, failed: bool = False):
    #     run_params = '-'.join(
    #         [f'{k}={v}' for k, v in self.frozen_run_params_v1.items()])
    #     path = os.path.join(result_root, run_params + '.err')
    #     if failed:
    #         path += ".failed"
    #     return path

    def build(self,
              src_root: str,
              build_root: str,
              force_rebuild: bool = False,
              dry_run: bool = False,
              logger: Optional[logging.Logger] = None):
        build_params = self.build_params
        backend = build_params.pop('backend')
        src_path = os.path.join(src_root, "backend_pim", backend)
        build_path = self.build_path(build_root=build_root)

        if backend == "cpu" or backend.endswith('@cpu'):
            logger.debug("==> Skipping building a CPU backend")
            return

        if not os.path.exists(build_path) or force_rebuild or dry_run:
            
            if not dry_run and force_rebuild:
                if os.path.exists(build_path):
                    shutil.rmtree(build_path)
            if logger is not None:
                logger.info(f"==> Building for {self}")
                logger.debug(f"==> backend: {backend}")
                logger.debug(f"==> src_path: {src_path}")
                logger.debug(f"==> build_path: {build_path}")
            try:
                build_lib(src_path=src_path,
                          build_path=build_path,
                          **build_params,
                          dry_run=dry_run,
                          logger=logger)
            except:
                if logger is not None:
                    logger.exception("Error occurred")
                if not dry_run:
                    shutil.rmtree(build_path)
        else:
            logger.debug(f"==> Skipping build at {build_path}")

    def status_at(self, result_root: str):
        stdout_path = self.stdout_path(result_root)
        if os.path.exists(stdout_path):
            return "done"
        if os.path.exists(stdout_path + ".failed"):
            return "failed"
        return "todo"




    def run(self,
            src_root: str,
            data_root: str,
            build_root: str,
            result_root: Optional[str] = None,
            repeat: int = 3,
            silent: bool = False,
            dry_run: bool = False,
            logger: Optional[logging.Logger] = None):
        stdout_path = None
        stderr_path = None
        if result_root is not None:
            stdout_path = self.stdout_path(result_root)
            stderr_path = self.stderr_path(result_root)
        backend = self.backend or self.default_backend

        # FIXME: We don't accept slash in params
        dataset = self.dataset if self.dataset != "ogbnproteins" else "ogbn-proteins"

        

        if backend in ("spmm_default", "spmm_multigroup"):
            version = "spmm"
        elif backend in ("spmm_grande",):
            version = "grande"
        elif backend in ("spmv_sparseP",):
            version = "spmv"
        elif backend in ("cpu", "CPU", None):
            version = "cpu"
        else:
            raise NotImplementedError(backend)

        if self.model is not None and self.num_layers is not None:
            test_py = f"{src_root}/inference.py"
        else:
            test_py = f"{src_root}/spmm_test.py"

        if (self.tune is not None) and self.tune:
            sp_ds_set = [(1, 32), (2, 16)]
            config = autotune(data_root, dataset, self.dense_size, sp_ds_set)
            self.sp_part, self.ds_part, self.balance, self.balance_tsklt, self.groups_per_rank = config

        if backend in ("cpu", "CPU", None):
            lib_path = None
        else:
            lib_path = self.build_path(build_root)
            lib_path = os.path.join(lib_path, "libbackend_pim.so")
            lib_path = '"' + lib_path + '"'

        cmd = f"python3 {test_py} " \
              f"--dataset={dataset} " \
              f"--datadir=\"{data_root}\" " \
              f"--sp_format={self.sp_format} " \
              f"--data_type={self.dtype} " \
              f"--hidden_size={self.dense_size} " \
              f"--sp_part={self.sp_part} " \
              f"--ds_part={self.ds_part} " \
              f"--repeat={repeat} " \
              f"--lib_path={lib_path} " \
              f"--version={version} "

        if self.nr_dpus is not None:
            cmd += f" --nr_dpus={self.nr_dpus}"

        # if self.tune is not None:
        #     cmd += f" --tune={self.tune}"

        if self.model is not None and self.num_layers is not None:
            cmd += f"--model={self.model} " \
                   f"--num_layers={self.num_layers} "

        # FIXME: we need to hack here because only multigroup backend accepts group_per_rank option
        if backend == "spmm_multigroup" and self.groups_per_rank is not None:
            cmd += f" --group_per_rank={self.groups_per_rank}"

        if logger is not None:
            logger.info(f"==> {self}")
            logger.debug(f"STDOUT: {stdout_path}")
            logger.debug(f"STDERR: {stderr_path}")

        if result_root is not None:
            result = run(cmd,
                         decode=False,
                         shell=True,
                         dry_run=dry_run,
                         logger=logger,
                         silent=silent)
            if not dry_run:
                stdout = result['stdout']
                stderr = result['stderr']
                if result['return'] != 0:
                    stdout_path = self.stdout_path(result_root, failed=True)
                    stderr_path = self.stderr_path(result_root, failed=True)
                with open(stdout_path, "wb") as writer:
                    writer.write(stdout)
                with open(stderr_path, "wb") as writer:
                    writer.write(stderr)
        else:
            run(cmd,
                decode=False,
                shell=True,
                logger=logger,
                silent=silent,
                dry_run=dry_run,
                capture_stderr=False,
                capture_stdout=False)

    def parse_result(self, result_root: str):
        result_path = self.stdout_path(result_root=result_root)
        engine = re.compile(r"^\[DATA](.*?): (.*)")
        results = collections.defaultdict(list)
        repeat = 0
        with open(result_path, "r") as reader:
            for line in reader:
                if line.startswith("-------------------- Repeat"):
                    repeat += 1
                if line.startswith("-------------------- Model"):
                    repeat += 1
                line = line.strip()
                parsed = engine.findall(line)
                if parsed:
                    key = parsed[0][0]
                    value = float(parsed[0][1])
                    results[key].append(value)
        for key, values in results.items():
            values = np.asarray(values).reshape(repeat, -1)
            values = values.mean(axis=0)  # mean on all repeat
            values = values.sum(axis=-1)  # sum on all fields within one repeat
            results[key] = values
        results['repeat'] = repeat
        return dict(results)
