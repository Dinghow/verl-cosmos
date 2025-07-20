# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the RoboVQA dataset to parquet format for video-language understanding
"""

import argparse
import os
import shutil

import datasets


_HDFS_PREFIX = "hdfs://"

def copy(src: str, dst: str, **kwargs) -> bool:
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    if os.path.isdir(src):
        return shutil.copytree(src, dst, **kwargs)
    else:
        return shutil.copy(src, dst, **kwargs)


def _mkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    os.makedirs(file_path, exist_ok=True)
    return True

def _is_non_local(path: str):
    return path.startswith(_HDFS_PREFIX)

def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)

FPS = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/robovqa")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Cosmos-Reason1-RL-Dataset/robovqa"
    train_dataset = datasets.load_dataset('../Cosmos-Reason1-RL-Datasetx10/robovqa', data_files='robovqa_rl_qa_pairs.json')['train']
    test_dataset = datasets.load_dataset('../Cosmos-Reason1-Benchmark/robovqa', data_files='robovqa_benchmark_qa_pairs.json')['train']
    # train_dataset = datasets.load_dataset('nvidia/Cosmos-Reason1-RL-Dataset', 'robovqa')['rl']
    # test_dataset = datasets.load_dataset('nvidia/Cosmos-Reason1-Benchmark', 'robovqa')['benchmark']


    user_prompt = "\nAnswer with the option's letter from the given choices directly."
    user_prompt += "\nPlease answer the question in the following format: <think> your reasoning </think> <answer> your answer </answer>."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # video = example.pop('video').split("/")[-1] # 使用pop移除原始字段
            if split == "train":
                video = os.path.join('data/Cosmos-Reason1-RL-Dataset/robovqa', example.pop('video'))
            else:
                video = os.path.join('data/Cosmos-Reason1-Benchmark/robovqa', example.pop('video'))
            qa_pairs = example.pop('qa_pairs')  # 使用pop移除原始字段
            answer = qa_pairs['answer']

            choices = qa_pairs['index2ans']
            problem = qa_pairs["question"] + "\n" + "\n".join([f"({i}) {choice}" for i, choice in choices.items()])
            prompt = f"<video>{problem}{user_prompt}"

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "videos": [{
                    "type": "video", 
                    "video": video,
                    "fps": FPS,
                }],
                "ability": "video_qa",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
