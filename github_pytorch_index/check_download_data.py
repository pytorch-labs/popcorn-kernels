# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

with open('/matx/u/simonguo/gh_triton_scrape/metadata.json', 'r') as f:
	try:
		data = json.load(f)
		print(f'JSON is well-formed')

		print('================================================')
		print('\nProject names:')
		for filename, project in data.items():
			print(f'- {project["full_name"]}')
		print('================================================')

		print(f'Number of entries: {len(data)}')

	except json.JSONDecodeError as e:
		print(f'JSON is malformed: {e}')
