# DDPO: Diversity-Driven Policy Optimization for Search-Based LLM Agents

[![Paper](https://img.shields.io/badge/Paper-Information%20Processing%20%26%20Management-2f6f9f)](https://doi.org/10.1016/j.ipm.2026.105104)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ipm.2026.105104-blue)](https://doi.org/10.1016/j.ipm.2026.105104)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

Official implementation of **“Restoring Exploration via Diversity-Driven Policy Optimization in Search-Based LLM Agents,”** accepted for publication in *Information Processing & Management*.

Xinyue Chen<sup>†</sup>, Pengyu Gao<sup>†</sup>, Jiangjiang Song, Liqian Chen, Kehang Zeng, Xin Yang, Xinjian Chen, and Xiaoyang Tan  
<sup>†</sup> Equal contribution.

- Paper: [https://doi.org/10.1016/j.ipm.2026.105104](https://doi.org/10.1016/j.ipm.2026.105104)
- Base framework: [Search-R1](https://github.com/PeterGriffinJin/Search-R1)

## Overview

Search-based LLM agents can learn when and how to retrieve external evidence through reinforcement learning. During training, however, their search behavior may collapse prematurely: different rollouts begin to issue similar queries and retrieve overlapping documents, reducing exploration and limiting the learning signal.

**Diversity-Driven Policy Optimization (DDPO)** augments group-based policy optimization with two complementary diversity signals:

- **Semantic Information Gain (SIG):** rewards semantic diversity among search queries.
- **Coverage Information Gain (CIG):** rewards diversity in the retrieved evidence.
- **Adaptive gating:** adjusts the strength of the diversity incentives according to the current rollout group, preserving stable policy optimization.

DDPO does not require an additional reward model or extra supervision. It is integrated into the Search-R1/veRL training pipeline and can be enabled or disabled through the training configuration.

## Highlights

- Restores exploration during reinforcement-learning fine-tuning of search-based LLM agents.
- Uses submodular log-determinant objectives to diversify queries and retrieved evidence.
- Adaptively gates exploration rewards to maintain stable and efficient convergence.
- Improves accuracy and search diversity across seven question-answering benchmarks.

The paper reports an average exact-match score of **52.1%** with a 7B backbone. On Natural Questions, DDPO reaches **44.8% EM** with **1.01 searches per question** on average; on MuSiQue, it improves Gold Passage Recall by **6.3%**.

## Repository Structure

~~~text
DDPO/
├── scripts/
│   ├── data_process/              # QA dataset preprocessing
│   ├── download.py                # Wiki-18 corpus and E5 index download
│   └── nq_hotpotqa/               # Reference evaluation scripts
├── search_r1/
│   ├── search/                    # Dense-retrieval server
│   └── utils/
├── verl/
│   ├── trainer/
│   │   ├── config/                # Training configuration
│   │   └── ppo/
│   │       └── improved_grpo_diversity.py
│   └── workers/
├── retrieval_launch.sh            # Retrieval-server launcher
├── train_grpo.sh                  # DDPO training launcher
├── LICENSE
└── NOTICE
~~~

The main DDPO implementation is in
<code>verl/trainer/ppo/improved_grpo_diversity.py</code>. Its integration with the training loop is in
<code>verl/trainer/ppo/ray_trainer.py</code>.

## Requirements

The code is intended for Linux systems with NVIDIA GPUs. The experiments in the paper used:

- Python 3.9 for training and Python 3.10 for retrieval;
- PyTorch 2.4.0 with CUDA 12.1;
- vLLM 0.6.3;
- one retrieval GPU in addition to the training GPUs;
- FlashAttention and FAISS-GPU.

The 3B and 7B experiments in the paper used 2×A100 and 8×H100 GPUs, respectively. Smaller local tests may use fewer GPUs after reducing the batch sizes, rollout parallelism, and the <code>trainer.n_gpus_per_node</code> setting.

## Installation

Clone the repository:

~~~bash
git clone https://github.com/TebooNok/DDPO.git
cd DDPO
~~~

Create the training environment:

~~~bash
conda create -n ddpo python=3.9 -y
conda activate ddpo

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "vllm==0.6.3" "transformers<4.48" "tensordict<0.6"
pip install accelerate codetiming datasets dill hydra-core numpy orjson pandas \
  pybind11 "ray[default]" tqdm wandb IPython matplotlib huggingface_hub
pip install flash-attn --no-build-isolation
~~~

Create a separate retrieval environment:

~~~bash
conda create -n ddpo-retriever python=3.10 -y
conda activate ddpo-retriever

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
pip install "transformers<4.48" datasets numpy tqdm uvicorn fastapi \
  pydantic huggingface_hub
~~~

Dependency compatibility can vary with the local CUDA driver and GPU type. If a binary package is unavailable for the system, install the matching PyTorch, vLLM, FlashAttention, and FAISS builds for the local CUDA runtime.

## Data Preparation

### 1. Question-answering datasets

The preprocessing script reads the unified datasets released by [FlashRAG](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets). Supported dataset names are:

- <code>nq</code>
- <code>triviaqa</code>
- <code>popqa</code>
- <code>hotpotqa</code>
- <code>2wikimultihopqa</code>
- <code>musique</code>
- <code>bamboogle</code>

For example, prepare Natural Questions:

~~~bash
python scripts/data_process/build_search_dataset.py \
  --dataset_name nq \
  --local_dir ./data/nq_search
~~~

This produces:

~~~text
data/nq_search/nq_train.parquet
data/nq_search/nq_test.parquet
~~~

The current <code>train_grpo.sh</code> expects files named <code>train.parquet</code> and <code>test.parquet</code>. Either update the two <code>data.*_files</code> entries in that script to use the generated filenames, or create copies with the expected names:

~~~bash
cp data/nq_search/nq_train.parquet data/nq_search/train.parquet
cp data/nq_search/nq_test.parquet data/nq_search/test.parquet
~~~

Run the preprocessing command separately for each additional benchmark. Bamboogle contains only a test split.

The preprocessing code uses the splits exposed by FlashRAG. In particular, it deterministically assigns the first 80% of the PopQA test split to training and the remaining 20% to testing; Bamboogle is test-only; and the development splits of HotpotQA, 2WikiMultiHopQA, and MuSiQue are used as test splits. Review
<code>scripts/data_process/build_search_dataset.py</code> before changing these conventions.

### 2. Wiki-18 retrieval corpus and E5 index

Download the corpus and index artifacts released with Search-R1:

~~~bash
mkdir -p dataset
python scripts/download.py --save_path ./dataset
~~~

Combine the two index parts and decompress the corpus:

~~~bash
cat dataset/part_aa dataset/part_ab > dataset/e5_Flat.index
gzip -dk dataset/wiki-18.jsonl.gz
~~~

The resulting files should be:

~~~text
dataset/e5_Flat.index
dataset/wiki-18.jsonl
~~~

The download sources are:

- [Wiki-18 corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)
- [Wiki-18 E5 index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index)

These are large artifacts: the two index parts total approximately 65 GB, and the compressed corpus is approximately 5 GB. Reserve additional disk space for the combined index and decompressed corpus, and ensure the retrieval machine has enough host memory to load the FAISS index. The current launcher loads the encoder on a GPU but does not pass <code>--faiss_gpu</code>, so the index remains in host memory by default.

## Running the Retriever

Open <code>retrieval_launch.sh</code> and set <code>file_path</code> to the absolute path of the prepared <code>dataset</code> directory. The default configuration uses [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2), returns the top three passages, and listens on port 8000.

Start the server from the repository root:

~~~bash
conda activate ddpo-retriever
bash retrieval_launch.sh
~~~

The training process expects the endpoint:

~~~text
http://127.0.0.1:8000/retrieve
~~~

Keep the retrieval server running while training or evaluating.

## Training

Before launching a run, edit <code>train_grpo.sh</code> and verify:

1. <code>CUDA_VISIBLE_DEVICES</code> matches the available training GPUs.
2. <code>DATA_DIR</code> points to the prepared dataset.
3. <code>BASE_MODEL</code> points to a local model or Hugging Face model ID.
4. <code>trainer.n_gpus_per_node</code> matches the number of visible training GPUs.
5. Batch sizes and micro-batch sizes fit the available GPU memory.
6. Proxy variables are removed or changed for the local network in both <code>train_grpo.sh</code> and <code>verl/utils/tracking.py</code>.
7. <code>WANDB_MODE</code> and <code>trainer.logger</code> match the desired logging mode; the launcher currently forces online W&B logging.
8. In <code>verl/trainer/main_ppo.py</code>, replace or remove the hard-coded Ray node address <code>10.0.0.2</code>. For a local single-node run, also consider setting <code>include_dashboard=False</code> and removing the public dashboard host.
9. The retriever URL and port match the running retrieval server.

The paper evaluates [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) and [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B). Access to model weights is subject to the model provider’s terms.

Start DDPO training:

~~~bash
conda activate ddpo
bash train_grpo.sh
~~~

The launcher writes actor checkpoints to
<code>verl_checkpoints/&lt;EXPERIMENT_NAME&gt;/actor/global_step_&lt;N&gt;</code> and logs to
<code>&lt;EXPERIMENT_NAME&gt;.log</code>. Weights & Biases logging is enabled by default. To disable it, edit the launcher’s <code>WANDB_MODE</code>/<code>trainer.logger</code> settings and remove the W&B-specific proxy assignments in <code>verl/utils/tracking.py</code>.

### Enabling or disabling DDPO

DDPO is controlled by:

~~~yaml
algorithm.use_diversity: true
~~~

Set it to <code>false</code> to use the base group-relative advantage without the diversity adjustment.

The default DDPO coefficients are defined in
<code>verl/trainer/ppo/improved_grpo_diversity.py</code>:

| Parameter | Default | Description |
|---|---:|---|
| <code>alpha_D</code> | 0.3 | Document-diversity contribution |
| <code>beta_E</code> | 0.4 | Exploration-reward scale |
| <code>zeta</code> | 0.5 | Adaptive-gating coefficient |
| <code>eta</code> | 0.5 | Adaptive-gating coefficient |
| <code>w_local</code> | 0.6 | Local diversity weight |
| <code>w_cross</code> | 0.4 | Cross-rollout diversity weight |
| <code>s_min</code> | 0.8 | Minimum adaptive scale |
| <code>s_max</code> | 1.3 | Maximum adaptive scale |

For faithful reproduction, start from the provided defaults and the hyperparameters documented in the paper.

## Evaluation

Evaluation uses the same retrieval service as training. The scripts under
<code>scripts/nq_hotpotqa/</code> provide a reference evaluation workflow.

Before running <code>scripts/nq_hotpotqa/evaluate.sh</code>:

- set <code>BASE_MODEL</code> inside the script to an actor checkpoint such as
  <code>verl_checkpoints/&lt;EXPERIMENT_NAME&gt;/actor/global_step_300</code>;
- set <code>DATA_DIR</code> to the evaluation Parquet files;
- adjust <code>CUDA_VISIBLE_DEVICES</code> and <code>trainer.n_gpus_per_node</code>;
- verify that the retriever is available at the configured URL.

Then run:

~~~bash
bash scripts/nq_hotpotqa/evaluate.sh
~~~

The provided evaluation launcher reports the task reward/exact-match score. The complete paper analysis—including searches per question, query-diversity measures, Gold Passage Recall, and aggregation over all seven benchmarks—requires the corresponding benchmark runs and analysis described in the paper; those metrics are not produced by this single script.

## Datasets and Resources

The experiments draw on the following public resources. Please cite the original dataset papers and comply with their individual licenses and terms:

| Resource | Link |
|---|---|
| FlashRAG unified datasets | [Hugging Face](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) · [GitHub](https://github.com/RUC-NLPIR/FlashRAG) |
| Natural Questions | [Official repository](https://github.com/google-research-datasets/natural-questions) |
| TriviaQA | [Official repository](https://github.com/mandarjoshi90/triviaqa) |
| PopQA | [Hugging Face](https://huggingface.co/datasets/akariasai/PopQA) |
| HotpotQA | [Official website](https://hotpotqa.github.io/) |
| 2WikiMultiHopQA | [Official repository](https://github.com/Alab-NII/2wikimultihop) |
| MuSiQue | [Official repository](https://github.com/StonyBrookNLP/musique) |
| Bamboogle / Self-Ask | [Official repository](https://github.com/ofirpress/self-ask) |
| KILT Wikipedia snapshot | [Official repository](https://github.com/facebookresearch/KILT) |
| Wiki-18 corpus | [Hugging Face](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) |
| Wiki-18 E5 index | [Hugging Face](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index) |

The repository does not redistribute the model weights or the datasets listed above.

## Citation

If this repository is useful in your research, please cite:

~~~bibtex
@article{chen2026restoring,
  title   = {Restoring Exploration via Diversity-Driven Policy Optimization in Search-Based LLM Agents},
  author  = {Chen, Xinyue and Gao, Pengyu and Song, Jiangjiang and Chen, Liqian and Zeng, Kehang and Yang, Xin and Chen, Xinjian and Tan, Xiaoyang},
  journal = {Information Processing \& Management},
  year    = {2026},
  pages   = {105104},
  doi     = {10.1016/j.ipm.2026.105104},
  url     = {https://doi.org/10.1016/j.ipm.2026.105104}
}
~~~

Because this codebase is derived from Search-R1, please also cite:

~~~bibtex
@inproceedings{jin2025searchr1,
  title     = {Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author    = {Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan O. and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  booktitle = {Proceedings of the Conference on Language Models},
  year      = {2025},
  url       = {https://openreview.net/forum?id=Rwhi91ideu}
}
~~~

## Acknowledgements

This repository is built upon and substantially adapted from
[Search-R1](https://github.com/PeterGriffinJin/Search-R1). We sincerely thank Bowen Jin and the Search-R1 contributors for releasing their code, retrieval resources, and training framework. Their work provided the foundation on which the DDPO implementation was developed.

We also thank the developers of [veRL](https://github.com/volcengine/verl) and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), as well as the creators and maintainers of the datasets and pretrained models used in this project.

## License

This repository is released under the [Apache License 2.0](LICENSE).

It contains modified code from Search-R1 and veRL. The original copyright and attribution notices are retained in [NOTICE](NOTICE). Third-party models, datasets, indexes, and other artifacts remain subject to their respective licenses and terms.

## Contact

For questions, reproducibility reports, or bug reports, please open a
[GitHub issue](https://github.com/TebooNok/DDPO/issues).
