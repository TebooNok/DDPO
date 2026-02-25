#!/usr/bin/env bash
# 构建 BGE-Large-en-v1.5 向量索引
set -e

# ============ 配置项（按需修改）============
corpus_file=/home/hwai/programs/Search-R1/dataset/wiki-18.jsonl
save_dir=/home/hwai/sjj/Search-R1/dataset                           # 知识库保存路径
retriever_name=bge-large-en-v1.5
retriever_model=/home/hwai/sjj/embedding_models/bge-large-en-v1.5

# FAISS 索引类型：Flat（精确检索，GPU加速）或 HNSW64（近似检索，CPU）
faiss_type=Flat
# ==========================================

echo "正在构建 ${retriever_name} 向量索引..."
echo "  语料库: ${corpus_file}"
echo "  模型路径: ${retriever_model}"
echo "  保存目录: ${save_dir}"
echo "  索引类型: ${faiss_type}"


CUDA_VISIBLE_DEVICES=3 python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 1024 \
    --pooling_method cls \
    --faiss_type $faiss_type \
    --save_embedding \

echo "✓ ${retriever_name} 索引构建完成！"
echo "  索引文件: ${save_dir}/${retriever_name}_${faiss_type}.index"


