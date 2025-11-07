# Transformer从零实现 - 中期作业

## 项目概述

本项目从零开始实现了完整的Transformer模型（包括Encoder和Decoder），在WikiText-2数据集上进行了语言建模任务的训练，
比较 绝对位置编码 与 T5 相对位置偏置 在 WikiText-2 语言建模任务上的性能。项目实现了原始论文中的所有核心组件，包括多头自注意力、位置编码、前馈网络等。



## 项目结构

```
BigModel/
├── src/
│ ├── init.py
│ ├── dataset.py # 数据加载与分词
│ ├── model.py # Transformer 模型与位置编码实现
│ ├── train.py # 主训练入口
│ ├── utils.py # 工具函数
├── results_abs/ # 绝对位置编码结果
│ ├── best_model.pt
│ ├── accuracy_curve.png, loss_curve.png, perplexity_curve.png, learning_rate_curve.png
│ └── attn_abs_epoch*.png
├── results_rel/ # 相对位置编码结果
│ ├── best_model.pt
│ ├── accuracy_curve.png, loss_curve.png, perplexity_curve.png, learning_rate_curve.png
│ └── attn_rel_epoch*.png
├── data/
│ └── wikitext-2.zip # 数据集（自动解压）
├── scripts/
│ └── run.sh # 一键运行脚本（包含两个实验）
├── requirements.txt
└── README.md
```


---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0.1 |
| CUDA | ≥ 12.0 |
| GPU | NVIDIA A100 / RTX 3090 (≥12GB VRAM 推荐) |
| 依赖库 | numpy, matplotlib |

安装方式：

conda create -n transformer python=3.10
conda activate transformer
pip install -r requirements.txt

## 项目结构
绝对位置编码实验
python src/train.py \
>   --data-archive data/wikitext-2.zip \
>   --save-dir results_rel \
>   --epochs 20 --batch-size 32 \
>   --d-model 256 --nhead 8 --enc-layers 3 --dec-layers 3 --d-ff 1024 \
>   --lr 3e-4 \
>   --use-relative-pos \
>   --rel-pos-num-buckets 32 \
>   --rel-pos-max-distance 128

相对位置编码实验
python src/train.py \
>   --data-archive data/wikitext-2.zip\
>   --save-dir results_abs \
>   --epochs 20 --batch-size 32 \
>   --d-model 256 --nhead 8 --enc-layers 3 --dec-layers 3 --d-ff 1024 \
>   --lr 3e-4
所有实验使用相同随机种子 42，可完全复现论文报告结果。

