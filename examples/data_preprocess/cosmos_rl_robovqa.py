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

import datasets
from verl.utils.hdfs_io import makedirs, copy

FPS = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/robovqa")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Cosmos-Reason1-RL-Dataset/robovqa"
    train_dataset = datasets.load_dataset('data/Cosmos-Reason1-RL-Dataset/robovqa', data_files='robovqa_rl_qa_pairs.json')['train']
    test_dataset = datasets.load_dataset('data/Cosmos-Reason1-Benchmark/robovqa', data_files='robovqa_benchmark_qa_pairs.json')['train']
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
