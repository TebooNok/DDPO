#!/usr/bin/env bash
# 启动 BGE-Large-en-v1.5 检索服务
# 端口：8000（如需并行多个服务，请改用不同端口并在 retrieval_server.py 增加 --port 支持）

set -e

# ============ 配置项 ============
index_file=/home/hwai/sjj/Search-R1/dataset/bge-large-en-v1.5_Flat.index    # 向量数据库地址
corpus_file=/home/hwai/programs/Search-R1/dataset/wiki-18.jsonl
retriever_name=bge-large-en-v1.5
retriever_model=/home/hwai/sjj/embedding_models/bge-large-en-v1.5
# ================================

echo "正在启动 ${retriever_name} 检索服务..."
echo "  索引: ${index_file}"
echo "  语料: ${corpus_file}"
echo "  模型: ${retriever_model}"
echo ""
echo "接口地址: http://127.0.0.1:8001/retrieve"
echo ""

python search_r1/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_model \
    --port 8001 \



