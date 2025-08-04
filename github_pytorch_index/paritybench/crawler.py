# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import json
import logging
import os
import re
import time

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


class CrawlGitHub(object):
    """
    Download projects from github with 100+ stars and the word "pytorch"
    """

    def __init__(self, download_dir, max_count=None, query=""):
        super(CrawlGitHub, self).__init__()
        self.download_dir = download_dir
        self.max_count = max_count  # max number of projects to download
        self.usr_query = query

    def github_search(self):
        base = "https://api.github.com/search/repositories?per_page=100&sort=stars"
        query = "pytorch+language:Python+stars:>100+size:<100000"
        if self.usr_query != "":
            query = self.usr_query

        seen = set()
        # both orders gets us 20 pages (past 10 limit), need 12 for current query
        for order in ("desc", "asc"):
            page = 1
            while True:
                time.sleep(6)  # https://developer.github.com/v3/search/#rate-limit
                rs = requests.get(f"{base}&page={page}&order={order}&q={query}")
                rs.raise_for_status()
                result = rs.json()
                assert not result["incomplete_results"]
                for project in result["items"]:
                    name = project["full_name"]
                    if self.max_count and len(seen) >= self.max_count:
                        return
                    if name not in seen:
                        seen.add(name)
                        yield project
                total_count = result["total_count"]
                log.info(
                    f"total_count={total_count} seen={len(seen)} page={page} {order}"
                )
                page += 1
                if (
                    len(result["items"]) == 0
                    or len(seen) >= total_count
                    or (self.max_count and len(seen) >= self.max_count)
                ):
                    return
                if page == 11:
                    break  # not allowed by API

    def get_projects_from_json(self, json_file: str, shard_num, shard_total):

        assert shard_num >= 0 and shard_total > shard_num

        project_list = []
        with open(json_file, "r") as fd:
            projects = json.load(fd)
            min_index = shard_num * len(projects) // shard_total
            max_index = (shard_num + 1) * len(projects) // shard_total
            max_index = min(len(projects), max_index)
            projects = projects[min_index:max_index]
        for project in projects:
            proj = self.get_project_dict(project)
            project_list.append(proj)
        return project_list

    def get_project_dict(self, project_json: dict[str, str]):
        repo_name = project_json["repo_name"]
        sha = project_json["sha"]
        project = {
            "full_name": repo_name,
            "html_url": f"https://github.com/{repo_name}",
            "default_branch": sha,
        }
        return project

    def download_project(self, project: dict):
        name = project["full_name"]
        url = project["html_url"]
        default_branch = project["default_branch"]
        output_filename = re.sub(r"[^a-zA-Z0-9]+", "_", name) + ".zip"
        output_path = os.path.join(self.download_dir, output_filename)
        # check if the file exists
        if os.path.exists(output_path):
            return output_filename
        try:
            rs = requests.get(f"{url}/archive/{default_branch}.zip", stream=True)
            rs.raise_for_status()
            with open(output_path, "wb") as fd:
                for chunk in rs.iter_content(chunk_size=8192):
                    fd.write(chunk)
            return output_filename
        except Exception as e:
            log.error(f"Failed to download {name} from {url}: {e}")
            return None

    def download(self, json_file=None, min_index=0, max_index=2000):
        metadata_path = os.path.join(self.download_dir, "metadata.json")
        if os.path.exists(metadata_path):
            print(f"Directory {self.download_dir} already exists. Skipping download.")
            return
        if not os.path.exists(self.download_dir):
            os.mkdir(self.download_dir)
        if json_file:
            projects = self.get_projects_from_json(json_file, min_index, max_index)
        else:
            projects = list(self.github_search())
        metadata = dict()
        for i, project in enumerate(projects):
            log.info(f"Downloading {project['full_name']} ({i + 1} of {len(projects)})")
            project_output_name = self.download_project(project)
            if project_output_name is not None:
                metadata[project_output_name] = project
        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)

    def download_parallel(
        self, num_workers=20, json_file=None, shard_num=0, shard_total=1
    ):
        """
        Download projects in parallel
        """
        print(f"Downloading projects in parallel with {num_workers} workers")

        metadata_path = os.path.join(self.download_dir, "metadata.json")

        os.path.exists(self.download_dir) or os.mkdir(self.download_dir)
        if json_file:
            projects = self.get_projects_from_json(json_file, shard_num, shard_total)
        else:
            projects = list(self.github_search())
        metadata = dict()

        def download_with_progress(args):
            i, project = args
            log.info(f"Downloading {project['full_name']} ({i + 1} of {len(projects)})")
            filename = self.download_project(project)
            return filename, project

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_project = {
                executor.submit(download_with_progress, (i, project)): project
                for i, project in enumerate(projects)
            }

            with tqdm(total=len(projects), desc="Downloading projects") as pbar:
                for future in concurrent.futures.as_completed(future_to_project):
                    try:
                        filename, project = future.result()
                        metadata[filename] = project
                        pbar.update(1)
                    except Exception as e:
                        log.error(f"Error downloading project: {e}")
                        pbar.update(1)

        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)
