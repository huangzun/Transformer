"""

本包提供了Transformer模型的完整实现，包括：
- 完整的Encoder-Decoder架构
- WikiText-2数据集的加载和处理
- 训练循环和评估函数
- 可视化和工具函数

使用示例：
    from src import Transformer, load_wikitext2, train_epoch
    from transformers import AutoTokenizer

    # 加载模型和数据
    model = Transformer(vocab_size=10000, d_model=128)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_texts, val_texts, test_texts = load_wikitext2(tokenizer)

    # 开始训练
    train_epoch(model, train_loader, optimizer, criterion, device)
"""

__version__ = "1.0.0"
__author__ = "Student Name"
__description__ = "Transformer Architecture Implementation from Scratch"

# 导入模型相关类
from .model import (
    Transformer,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    MultiHeadAttention,
    ScaledDotProductAttention,
    FeedForwardNetwork,
    PositionalEncoding
)

# 导入数据集相关类和函数
from .dataset import (
    WikiText2Dataset,
    SequenceToSequenceDataset,
    load_wikitext2,
    load_iwslt2017,
    create_dataloaders
)

# 导入训练相关函数
from .train import (
    train_epoch,
    evaluate,
    plot_results
)

# 导入工具函数
from .utils import (
    create_masks,
    create_causal_mask,
    create_combined_mask,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    plot_training_curves,
    visualize_attention,
    calculate_perplexity,
    set_seed,
    get_device,
    print_model_info
)

# 定义公开API
__all__ = [
    # 模型类
    'Transformer',
    'Encoder',
    'Decoder',
    'EncoderLayer',
    'DecoderLayer',
    'MultiHeadAttention',
    'ScaledDotProductAttention',
    'FeedForwardNetwork',
    'PositionalEncoding',

    # 数据集类和函数
    'WikiText2Dataset',
    'SequenceToSequenceDataset',
    'load_wikitext2',
    'load_iwslt2017',
    'create_dataloaders',

    # 训练函数
    'train_epoch',
    'evaluate',
    'plot_results',

    # 工具函数
    'create_masks',
    'create_causal_mask',
    'create_combined_mask',
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint',
    'plot_training_curves',
    'visualize_attention',
    'calculate_perplexity',
    'set_seed',
    'get_device',
    'print_model_info',
]


def get_info():
    """获取包的信息"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': ['model', 'dataset', 'train', 'utils']
    }


if __name__ == '__main__':
    print(f"Transformer Package v{__version__}")
    print(f"Available modules: {', '.join(__all__)}")