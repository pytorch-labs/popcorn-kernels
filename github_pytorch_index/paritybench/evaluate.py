# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import threading
import time
from functools import partial
from multiprocessing.pool import ThreadPool

import pandas as pd
import torch
import torch._dynamo
import torch._inductor

from paritybench.reporting import ErrorAggregatorDict, Stats
from paritybench.utils import (
	get_cosine_and_fp64_outputs,
	get_skiplist,
	get_tol,
	patch_torch_manual_seed,
	reset_rng_state,
	subproc_wrapper,
	wrap_args,
)
from run_and_check import import_Model_and_args_from_code
from torch._dynamo.testing import same
from torch.testing._internal.jit_utils import JitTestCase

from tqdm import tqdm


log = logging.getLogger(__name__)

# Remove inductor randomness
torch._inductor.config.fallback_random = True
# Remove randomeness when torch manual seed is called
patch_torch_manual_seed()

lock = threading.Lock()


class EagerFailed(RuntimeError):
	pass


class OnnxFailed(RuntimeError):
	pass


class JitFailed(RuntimeError):
	pass


def set_env_vars(log_name, cache_folder):
	# if we're in a tmux session, set the env vars local to the session so we can parallize
	# if "TMUX" in os.environ:
	#     cmd_logs = f"tmux set-environment TORCH_LOGS_OUT {log_name}"
	#     cmd_cache = f"tmux set-environment TORCHINDUCTOR_CACHE_DIR {cache_folder}"
	#     subprocess.run(cmd_logs, shell=True)
	#     subprocess.run(cmd_cache, shell=True)
	# else:
	os.environ['TORCH_LOGS_OUT'] = log_name
	os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_folder
	torch._logging.set_logs(output_code=True)


def evaluate_nn_module(
	nn_cls,
	get_init_args,
	get_forward_args,
	record_error,
	main_args,
	repo_name,
	run_dir: str = None,
):
	"""
	Run an nn.Module with torch.jit.script and see if it works the same
	as eager.

	:param nn_cls: a subclass of nn.Module to be tested
	:param get_init_args: function that returns (args, kwargs)
	:param get_forward_args: function that returns (args, kwargs)
	:param record_error: function to record an exception for debugging/reporting
	:return: True if the test passes
	"""

	# TORCHINDUCTOR_UNIQUE_KERNEL_NAMES as we need it to extract triton code on certain python versions
	os.environ['TORCHINDUCTOR_UNIQUE_KERNEL_NAMES'] = '1'
	try:
		args, kwargs = get_init_args()
		nn = nn_cls(*args, **kwargs)
	except Exception as e:
		record_error('init', e)
		raise EagerFailed()

	device = torch.device(main_args.device)
	try:
		nn.eval()
		nn.to(device)
	except Exception:
		pass

	nn_script = None
	if main_args.compile_mode == 'torchscript':
		try:
			nn_script = torch.jit.script(nn)
		except Exception as e:
			record_error('compile {}'.format(main_args.compile_mode), e)
			raise JitFailed()

	is_inductor_test = (
		main_args.compile_mode in ('dynamo', 'aot_inductor') and main_args.backend == 'inductor'
	)
	is_export_test = main_args.compile_mode == 'export'

	if (is_inductor_test or is_export_test) and run_dir:
		log_dir = os.path.join(run_dir, 'inductor_logs')
		cache_dir = os.path.join(run_dir, 'inductor_cache')
		os.makedirs(log_dir, exist_ok=True)
		os.makedirs(cache_dir, exist_ok=True)
		log_name = os.path.join(run_dir, f'inductor_logs/{repo_name}.{nn_cls.__name__}.txt')

		set_env_vars(log_name, cache_dir)

	cosine = False
	fp64_outputs = None

	try:
		args = get_forward_args()
		args = wrap_args(args, device)

		if is_inductor_test:
			cosine, fp64_outputs = get_cosine_and_fp64_outputs(nn, args, kwargs={})

		if main_args.metric_path:
			torch.cuda.synchronize()
			eager_start_ts = time.perf_counter()
		# The first eager run
		reset_rng_state()
		result1 = nn(*args)
		if main_args.metric_path:
			torch.cuda.synchronize()
			eager_elapse = time.perf_counter() - eager_start_ts

		# The second eager run
		reset_rng_state()
		result2 = nn(*args)
	except Exception as e:
		record_error('run_eager', e)
		raise EagerFailed()

	if main_args.onnxdir:
		try:
			onnx_path = '{}/{}.onnx'.format(main_args.onnxdir, nn_cls.__name__)
			torch.onnx.export(nn, *args, onnx_path)
		except Exception as e:
			record_error('export_onnx', e)
			raise OnnxFailed()

	if main_args.metric_path:
		torch.cuda.synchronize()
		dynamo_start_ts = time.perf_counter()

	try:
		if nn_script:
			result3 = nn_script(*args)
		else:
			# Dynamo/Inductor/Export run
			reset_rng_state()
			torch._dynamo.reset()
			if main_args.compile_mode == 'dynamo':
				compiled_model = torch._dynamo.optimize(
					main_args.backend, nopython=main_args.fullgraph
				)(nn)
				result3 = compiled_model(*args)
			elif main_args.compile_mode == 'export':
				exported_model = torch.export.export(
					nn,
					tuple(args),
					strict=False,
				).module()
				result3 = exported_model(*args)
			elif main_args.compile_mode == 'aot_inductor':
				ep = torch.export.export(
					nn,
					tuple(args),
					strict=False,
				)
				ap = torch._inductor.aoti_compile_and_package(ep)
				compiled_model = torch._inductor.aoti_load_package(ap)
				result3 = compiled_model(*args)
			else:
				raise AssertionError('Invalid compile_mode')

	except Exception as e:
		record_error('run_jit {} '.format(main_args.compile_mode), e)
		raise JitFailed()

	if main_args.metric_path:
		torch.cuda.synchronize()
		dynamo_elapse = time.perf_counter() - dynamo_start_ts

	tol = get_tol(main_args)
	try:
		JitTestCase().assertEqual(result1, result2)
		try:
			# Dynamo/Inductor/Export accuracy check against eager mode
			if is_inductor_test:
				JitTestCase().assertTrue(
					same(
						result2,
						result3,
						fp64_ref=fp64_outputs,
						cos_similarity=cosine,
						tol=tol,
					)
				)
			else:
				JitTestCase().assertEqual(result2, result3, atol=tol, rtol=tol)
		except Exception as e:
			record_error('check_output', e)
			raise JitFailed()
	except AssertionError:
		pass  # output is not deterministic, cant check it -- assuming correct

	# Record compilation metrics
	if main_args.metric_path:
		from torch._dynamo.utils import compilation_metrics

		model_id = f'{nn_cls.__module__}.{nn_cls.__name__}'
		compilation_metrics = {
			'model_id': model_id,
			'dynamo_wall_time': dynamo_elapse,
			'eager_wall_time': eager_elapse,
			'wall_time_diff': dynamo_elapse - eager_elapse,
			'_compile': compilation_metrics.get('_compile', [0.0])[0],
		}

		with lock, open(main_args.metric_path, 'a') as f:
			logline = []
			for _, v in compilation_metrics.items():
				if isinstance(v, float):
					logline.append(f'{v:.3f}')
				else:
					logline.append(str(v))
			f.write(' '.join(logline))
			f.write('\n')

	return True


def evaluate_pyfile_subproc(tempdir: str, path: str, args, run_dir: str = None):
	"""
	Evaluate/test all the TESTCASES in path.

	:param path: *.py file to test
	:return: errors, stats
	"""
	errors = ErrorAggregatorDict(path)
	stats = Stats()
	module_name = path.split('/')[-1].split('.')[1]
	code = open(path, 'r').read()
	repo_name = path.split('/')[-1].split('.')[0]
	nn_cls, get_forward_args, get_init_args = import_Model_and_args_from_code(code, module_name)

	if not nn_cls:
		return errors, stats

	stats['projects'] += 1

	index = -1
	skiplist_name = f'{path}:{nn_cls.__name__}'
	if args.filter and args.filter not in nn_cls.__name__:
		stats['tests_skipped'] += 1
	elif (get_skiplist(args)) and (skiplist_name in get_skiplist(args)):
		stats['tests_skipped'] += 1
	elif nn_cls.forward.__name__ == '_forward_unimplemented':
		stats['tests_skipped'] += 1
	else:
		stats['tests'] += 1
		repro = f'{nn_cls.__name__} # pytest {path} -k test_{index:03d}'
		try:
			rv = evaluate_nn_module(
				nn_cls,
				get_init_args,
				get_forward_args,
				partial(errors.record, module=repro),
				main_args=args,
				repo_name=repo_name,
				run_dir=run_dir,
			)
			stats['tests_passed'] += int(rv)
		except JitFailed:
			pass
		except EagerFailed:
			stats['eager_failed'] += 1
		except OnnxFailed:
			pass

	stats['tests'] = stats['tests'] - stats['eager_failed']
	stats['tests_failed'] = stats['tests'] - stats['tests_passed']

	if not stats['tests']:
		# eager failed not the jit, remove from totals
		stats['projects'] -= 1
	elif stats['tests_failed']:
		stats['projects_failed'] += 1
	else:
		stats['projects_passed'] += 1

	return errors, stats


def evaluate_all(
	args,
	run_dir: str = './run/run1',
	offset: int = 0,
	limit: int = None,
	jobs=4,
):
	"""
	Generate a paritybench score, main entrypoint for this module.

	:param run_dir: directory which contains all the artifacts for the run
	:param limit: optional maximum number of files to process
	:param fn: inner function to run the tests
	:param jobs: how many processes to run at once
	"""

	assert args.shard_num < args.shard_total, 'shard_num must be less than shard_total'

	tests_dir = os.path.join(run_dir, 'cleaned_pytorch_modules')
	synthetic_tests_dir = os.path.join(run_dir, 'synthetic_modules')
	os.makedirs(tests_dir, exist_ok=True)
	os.makedirs(synthetic_tests_dir, exist_ok=True)

	feval = partial(evaluate_pyfile_subproc, args=args, run_dir=run_dir)
	fn = partial(subproc_wrapper, fn=feval, fresh_cache_dir=args.fresh_cache_dir)
	start = time.time()
	stats = Stats()
	errors = ErrorAggregatorDict()
	testfiles = [os.path.join(tests_dir, f) for f in os.listdir(tests_dir)]

	if args.synthetic_data_dir:
		synthetic_files = [f for f in os.listdir(args.synthetic_data_dir) if f.endswith('.py')]
		synthetic_files.sort()
		if args.shard_num > len(synthetic_files):
			synthetic_files = []
		elif args.shard_total > 1:
			synthetic_files = [
				f for i, f in enumerate(synthetic_files) if i % args.shard_total == args.shard_num
			]
		for f in synthetic_files:
			file_code = open(os.path.join(args.synthetic_data_dir, f), 'r').read()
			# we do this rename for 1) logging and 2) to keep the same pipeline / structure
			synthetic_file = os.path.join(synthetic_tests_dir, f'POPCORN_SYNTHETIC_DATA.{f}')
			with open(synthetic_file, 'w') as f:
				f.write(file_code)
			testfiles.append(synthetic_file)

	testfiles.sort()

	if limit:
		testfiles = testfiles[offset : offset + limit]

	pool = ThreadPool(jobs)
	for errors_part, stats_part in tqdm(
		pool.imap_unordered(fn, testfiles),
		total=len(testfiles),
		desc='Evaluating tests',
	):
		errors.update(errors_part)
		stats.update(stats_part)
	pool.close()
	errors.print_report()
	index = ('projects', 'tests')
	report = pd.DataFrame(
		[
			[
				stats[f'{k}'],
				stats[f'{k}_passed'],
				'{:.1%}'.format(stats[f'{k}_passed'] / (stats[f'{k}'] or 1)),
			]
			for k in index
		],
		index=index,
		columns=['total', 'passing', 'score'],
	)

	log.info(
		f'TOTAL: {stats}, took {time.time() - start:.1f} seconds\n\n{args.compile_mode} {args.backend} ParityBench:\n{report}'
	)
