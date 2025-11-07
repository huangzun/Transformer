
---

## ✅ 配套脚本 `scripts/run.sh`

```bash
#!/bin/bash
# ===============================================
# Transformer 从零实现 - 一键运行脚本
# Author: 黄尊爱 (Student ID: 25125379)
# ===============================================

set -e

echo "==== [1/2] 运行绝对位置编码实验 ===="
python src/train.py \
  --data-archive data/wikitext-2.zip \
  --save-dir results_rel \
  --epochs 20 --batch-size 32 \
  --d-model 256 --nhead 8 --enc-layers 3 --dec-layers 3 --d-ff 1024 \
  --lr 3e-4 \
  --use-relative-pos \
  --rel-pos-num-buckets 32 \
  --rel-pos-max-distance 128

echo "==== [2/2] 运行相对位置编码实验 ===="
python src/train.py \
  --data-archive data/wikitext-2.zip\
  --save-dir results_abs \
  --epochs 20 --batch-size 32 \
  --d-model 256 --nhead 8 --enc-layers 3 --dec-layers 3 --d-ff 1024 \
  --lr 3e-4

echo "✅ 所有实验完成！结果保存在 results_abs/ 与 results_rel/ 目录。"

