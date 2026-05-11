#!/bin/bash
# CIMA 全流程运行脚本
# 用法: cd CIMA && bash scripts/run_all.sh v3

set -e

VERSION=${1:-v3}
DATASET="data/dataset.xlsx"
OUTPUT_BASE="results"

echo "=========================================="
echo "  CIMA Full Pipeline - Version: $VERSION"
echo "=========================================="

SEEN_SET="data/splits/$VERSION/seen_set.xlsx"
UNSEEN_SET="data/splits/$VERSION/unseen_test_set.xlsx"
OUTPUT_DIR="$OUTPUT_BASE/$VERSION"

# Step 1: 数据划分（如已有 splits 则跳过）
if [ ! -f "$SEEN_SET" ]; then
    echo "[Step 1] 数据划分..."
    python data/split.py --dataset "$DATASET" --output-dir data/splits --versions "$VERSION"
else
    echo "[Step 1] 已有划分数据，跳过"
fi

# Step 2: 预计算 Embedding (FMem)
if [ ! -f "data/global_embeddings.npy" ]; then
    echo "[Step 2] 预计算 Embedding..."
    python -m cima.fmem.precompute_embeddings --dataset "$DATASET" --output-dir data
else
    echo "[Step 2] 已有 Embedding，跳过"
fi

# Step 3: COAT 因子发现 (CMem)
if [ ! -f "$OUTPUT_DIR/coat_factors.json" ]; then
    echo "[Step 3] COAT 因子发现..."
    python -m cima.cmem.coat_discovery --seen-set "$SEEN_SET" --output-dir "$OUTPUT_DIR"
else
    echo "[Step 3] 已有因子文件，跳过"
fi

# Step 4: 推理实验
echo "[Step 4] 推理实验..."

echo "  [4a] Zero-shot..."
python -m cima.inference --mode zero_shot \
    --seen-set "$SEEN_SET" --unseen-set "$UNSEEN_SET" \
    --output-dir "$OUTPUT_DIR"

echo "  [4b] FMem-only..."
python -m cima.inference --mode fmem_only \
    --seen-set "$SEEN_SET" --unseen-set "$UNSEEN_SET" \
    --embeddings data/global_embeddings.npy \
    --embed-meta data/global_embed_meta.json \
    --output-dir "$OUTPUT_DIR"

echo "  [4c] CMem-only..."
python -m cima.inference --mode cmem_only \
    --seen-set "$SEEN_SET" --unseen-set "$UNSEEN_SET" \
    --coat-factors "$OUTPUT_DIR/coat_factors.json" \
    --output-dir "$OUTPUT_DIR"

echo "  [4d] CMem+FMem..."
python -m cima.inference --mode cmem_fmem \
    --seen-set "$SEEN_SET" --unseen-set "$UNSEEN_SET" \
    --coat-factors "$OUTPUT_DIR/coat_factors.json" \
    --embeddings data/global_embeddings.npy \
    --embed-meta data/global_embed_meta.json \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "  All experiments completed!"
echo "  Results saved to: $OUTPUT_DIR/"
echo "=========================================="
