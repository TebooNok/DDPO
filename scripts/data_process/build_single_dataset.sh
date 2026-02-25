#!/usr/bin/env bash
# 构建单个数据集
# 使用方法: bash scripts/data_process/build_single_dataset.sh triviaqa

set -e

if [ -z "$1" ]; then
    echo "错误: 未指定数据集名称"
    echo "使用方法: bash scripts/data_process/build_single_dataset.sh <dataset_name>"
    echo ""
    echo "可用的数据集:"
    echo "  - nq"
    echo "  - triviaqa"
    echo "  - popqa"
    echo "  - hotpotqa"
    echo "  - 2wikimultihopqa"
    echo "  - musique"
    echo "  - bamboogle"
    exit 1
fi

DATASET_NAME=$1

echo "========================================"
echo "构建数据集: $DATASET_NAME"
echo "========================================"
echo ""

python scripts/data_process/build_search_dataset.py --dataset_name "$DATASET_NAME"

echo ""
echo "✓ 数据集构建完成！"
echo "  位置: ./data/${DATASET_NAME}_search/"
echo "  文件: ${DATASET_NAME}_train.parquet, ${DATASET_NAME}_test.parquet"

