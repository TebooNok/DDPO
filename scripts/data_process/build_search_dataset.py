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
通用的数据集预处理脚本，支持多个 FlashRAG 数据集
"""

import os
import datasets
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


def process_dataset(dataset_name, local_dir, template_type='base'):
    """
    处理单个数据集
    
    Args:
        dataset_name: 数据集名称，如 'nq', 'triviaqa', 'popqa' 等
        local_dir: 本地保存目录
        template_type: prompt 模板类型
    """
    print(f"正在处理数据集: {dataset_name}")
    print(f"保存路径: {local_dir}")
    
    # 加载数据集
    print(f"从 RUC-NLPIR/FlashRAG_datasets 加载 {dataset_name}...")
    try:
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', dataset_name)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return False
    
    print(f"数据集可用分割: {list(dataset.keys())}")
    
    # 根据不同数据集的分割情况进行处理
    if dataset_name == 'popqa':
        # popqa 只有 test，按 8:2 分割为 train 和 test
        print("popqa 数据集只有 test，按 8:2 分割为 train 和 test")
        full_test = dataset['test']
        # 计算分割点
        total_size = len(full_test)
        train_size = int(total_size * 0.8)
        
        # 使用 select 分割数据集
        train_dataset = full_test.select(range(train_size))
        test_dataset = full_test.select(range(train_size, total_size))
        
    elif dataset_name == 'bamboogle':
        # bamboogle 只有 test，只保存 test
        print("bamboogle 数据集只有 test，仅保存测试集")
        test_dataset = dataset['test']
        train_dataset = None
        
    elif dataset_name in ['hotpotqa', '2wikimultihopqa', 'musique']:
        # 这些数据集有 train 和 dev，将 dev 作为 test
        print(f"{dataset_name} 数据集有 train 和 dev，将 dev 作为 test")
        train_dataset = dataset['train']
        test_dataset = dataset['dev']
        
    else:
        # 标准情况：有 train 和 test
        train_dataset = dataset['train']
        test_dataset = dataset['test']
    
    if train_dataset is not None:
        print(f"✓ 训练集大小: {len(train_dataset)}")
    else:
        print("✓ 无训练集（仅测试集）")
    print(f"✓ 测试集大小: {len(test_dataset)}")

    # 定义映射函数
    def make_map_fn(split):
        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": dataset_name,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    # 应用映射
    if train_dataset is not None:
        print("正在处理训练集...")
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    print("正在处理测试集...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # 保存为 parquet
    os.makedirs(local_dir, exist_ok=True)
    
    if train_dataset is not None:
        train_path = os.path.join(local_dir, f'{dataset_name}_train.parquet')
        print(f"保存训练集到: {train_path}")
        train_dataset.to_parquet(train_path)
    
    test_path = os.path.join(local_dir, f'{dataset_name}_test.parquet')
    print(f"保存测试集到: {test_path}")
    test_dataset.to_parquet(test_path)
    
    print(f"✓ {dataset_name} 数据集处理完成！\n")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="处理 FlashRAG 数据集")
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='数据集名称，如: nq, triviaqa, popqa, hotpotqa 等')
    parser.add_argument('--local_dir', type=str, default=None,
                        help='保存目录，默认为 ./data/{dataset_name}_search')
    parser.add_argument('--template_type', type=str, default='base',
                        help='Prompt 模板类型')

    args = parser.parse_args()
    
    # 如果未指定保存目录，使用默认格式
    if args.local_dir is None:
        args.local_dir = f'./data/{args.dataset_name}_search'
    
    success = process_dataset(
        dataset_name=args.dataset_name,
        local_dir=args.local_dir,
        template_type=args.template_type
    )
    
    if success:
        print("=" * 50)
        print(f"✓ 数据集 {args.dataset_name} 已成功构建！")
        if args.dataset_name != 'bamboogle':
            print(f"  训练集: {args.local_dir}/{args.dataset_name}_train.parquet")
        print(f"  测试集: {args.local_dir}/{args.dataset_name}_test.parquet")
        if args.dataset_name == 'bamboogle':
            print(f"  注意: bamboogle 仅包含测试集")
        print("=" * 50)
    else:
        print(f"❌ 数据集 {args.dataset_name} 处理失败")
        exit(1)

