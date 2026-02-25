#!/usr/bin/env bash
# 启动 Qwen3-Embedding-0.6B 检索服务
# 端口：8002

set -e

# ============ 配置项 ============
index_file=/home/hwai/sjj/Search-R1/dataset/qwen3-embedding-0.6b_Flat.index # 向量知识库位置
corpus_file=/home/hwai/programs/Search-R1/dataset/wiki-18.jsonl
retriever_name=Qwen3-Embedding-0.6B
retriever_model=/home/hwai/sjj/embedding_models/Qwen3-Embedding-0.6B
# ================================

echo "正在启动 ${retriever_name} 检索服务..."
echo "  索引: ${index_file}"
echo "  语料: ${corpus_file}"
echo "  模型: ${retriever_model}"
echo ""
echo "接口地址: http://127.0.0.1:8002/retrieve"
echo ""

python search_r1/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_model \
    --port 8002 \


