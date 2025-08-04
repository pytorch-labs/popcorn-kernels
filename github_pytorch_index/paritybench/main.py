# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
from functools import partial

import torch
import torch._dynamo

from paritybench.crawler import CrawlGitHub
from paritybench.evaluate import evaluate_all, evaluate_pyfile_subproc
from paritybench.generate import generate_all, generate_zipfile_subproc, write_helpers
from paritybench.utils import subproc_wrapper, tempdir_wrapper

log = logging.getLogger(__name__)


def main_one_file(fn, path, args):
	if ':' in path and not args.filter:
		path, args.filter = path.split(':', 2)
	assert os.path.isfile(path) or os.path.isdir(path)

	fn = partial(fn, args=args)

	if not args.no_fork:
		wrapper = subproc_wrapper
	else:
		wrapper = tempdir_wrapper

	errors, stats = wrapper(path, fn=fn, fresh_cache_dir=args.fresh_cache_dir)

	errors.print_report()
	log.info(f'Stats: {stats}')
	return


def get_args(raw_args=None):
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	group.add_argument(
		'--download',
		action='store_true',
		help='[SLOW:days] crawl and download top github projects',
	)
	parser.add_argument(
		'--repos_file',
		type=str,
		help='file containing list of github repos to download',
		default=None,
	)
	parser.add_argument(
		'--shard_num',
		type=int,
		help='shard number of the current process. This is ignored if repos file is not given or evaluate-all (with a synthetic directory) is not called',
		default=0,
	)
	parser.add_argument(
		'--shard_total',
		type=int,
		help='total number of shards. This is ignored if repos file is not given or evaluate-all (with a synthetic directory) is not called',
		default=1,
	)

	group.add_argument(
		'--generate-all',
		action='store_true',
		help='Turn crawled github projects into generated testcases',
	)
	group.add_argument('--generate-one', '-g', help='Process a .zip file from a github download')
	parser.add_argument(
		'--generate_chunk_num',
		type=int,
		help='For --generate-all, the chunk number to process (needs generate_num_chunks)',
	)
	parser.add_argument(
		'--generate_num_chunks',
		type=int,
		help='For --generate-all, the total number of chunks to process (needs generate_chunk_num)',
	)

	group.add_argument(
		'--evaluate-one', '-e', help='Check torch.jit.script on a given test_*.py file'
	)
	group.add_argument('--evaluate-all', action='store_true', help='Check torch.jit.script parity')

	# add a flag for parallel download
	parser.add_argument(
		'--parallel-download', action='store_true', help='Download projects in parallel'
	)

	# TODO: Sahan, put everything (donwload, build, generate, cache) in all one
	parser.add_argument(
		'--run-dir',
		default='./runs/run1',
		help='dir where we have all artifacts for the run (intermediate outputs (ie. download, build, generate, cache) + final dataset)',
	)

	# Number of Parallel Jobs
	parser.add_argument('--jobs', '-j', type=int, default=4)
	parser.add_argument(
		'--offset',
		type=int,
		default=0,
		help='Pick files starting from this offset. Together with --limit, we can run through all files in multiple separate runs',
	)
	parser.add_argument('--limit', '-l', type=int, help='only run the first N files')
	parser.add_argument('--filter', '-f', '-k', help='only run module containing given name')
	parser.add_argument(
		'--no-fork', action='store_true', help="don't run *-one test in a subprocess"
	)
	parser.add_argument('--memory-limit-gb', type=int, default=32)

	parser.add_argument(
		'--onnxdir',
		type=str,
		help='dir where to export modules to onnx during evaluate',
	)
	parser.add_argument(
		'--fullgraph',
		default=False,
		action='store_true',
		help='use fullgraph(no python fall back) when compiling with dynamo',
	)
	parser.add_argument(
		'--compile_mode',
		default='dynamo',
		type=str,
		help='choose a mode of compilation: dynamo, export, aot_inductor or torchscript',
	)
	parser.add_argument(
		'--backend',
		default='inductor',
		type=str,
		help='dynamo backends: {}'.format(torch._dynamo.list_backends()),
	)
	parser.add_argument(
		'--device', default='cuda', type=str, help='evaluate modules using cuda or cpu'
	)
	parser.add_argument('--metric-path', type=str, help='path of the compilation metric')
	parser.add_argument(
		'--fresh-cache-dir',
		action='store_true',
		help='use a fresh cache dir for each individual inductor test run and remove it after done',
	)
	parser.add_argument(
		'--synthetic-data-dir',
		type=str,
		help='path to the synthetic data directory. This is only used for --evaluate-all',
	)
	args = parser.parse_args(raw_args)
	return args


def main(raw_args=None):
	assert sys.version_info >= (3, 8), 'Python 3.8+ required, got: {}'.format(sys.version)
	logging.basicConfig(level=logging.INFO)
	args = get_args(raw_args)

	download_dir = args.run_dir + '/download'

	# create directories if they don't exist
	os.makedirs(args.run_dir, exist_ok=True)
	if args.download:
		os.makedirs(download_dir, exist_ok=True)

	os.environ['RLIMIT_AS_GB'] = str(args.memory_limit_gb)

	if args.download:
		if args.parallel_download:
			print('Downloading projects in parallel')
			return CrawlGitHub(download_dir, max_count=args.limit).download_parallel(
				args.jobs, args.repos_file, args.shard_num, args.shard_total
			)
		else:
			return CrawlGitHub(download_dir, max_count=args.limit).download()

	write_helpers(args.run_dir)
	# generate mode doesn't work well with `spawn`
	if not args.generate_one and not args.generate_all:
		torch.multiprocessing.set_start_method('spawn')

	if args.evaluate_one:
		return main_one_file(evaluate_pyfile_subproc, args.evaluate_one, args)

	if args.generate_one:
		return main_one_file(generate_zipfile_subproc, args.generate_one, args)

	if args.generate_all:
		return generate_all(
			args,
			download_dir=download_dir,
			limit=args.limit,
			jobs=args.jobs,
			chunk_num=args.generate_chunk_num,
			num_chunks=args.generate_num_chunks,
		)

	# args.evaluate_all is the default:
	return evaluate_all(
		args,
		run_dir=args.run_dir,
		offset=args.offset,
		limit=args.limit,
		jobs=args.jobs,
	)
