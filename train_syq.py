from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs
import pandas as pd
import numpy as np
import random
import torch

# 定义一个函数来设置所有相关模块的随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 设置随机种子，确保训练结果一致
set_seed(3407)

# 读取特征文件
features_data = pd.read_csv('BCF_with_descriptors.csv')

# 打印特征文件的形状和前几行
print(f"特征文件的形状: {features_data.shape}")
print(features_data.head())

# 定义训练参数
train_args = TrainArgs().parse_args([
    '--data_path', 'BCF.csv',  # 数据文件路径
    '--features_path', 'BCF_with_descriptors.csv',  # RDKit 描述符特征文件路径
    '--dataset_type', 'regression',  # 任务类型为回归0
    '--save_dir', 'chemprop_checkpoints',  # 模型保存路径
    '--target_columns', 'target',  # 要预测的列名
    '--smiles_column', 'smiles',  # SMILES 列名
    '--epochs', '50',  # 训练轮数
    '--batch_size', '32',  # 每批样本数量
    '--hidden_size', '1007',  # GNN的输出维数
    '--depth', '2',  # GNN的层数
    '--ffn_hidden_size', '151',  # 前馈网络的输出维数
    '--ffn_num_layers', '1',  # 前馈网络的层数
    '--dropout', '0.30703855324064994',  # Dropout率
    '--split_type', 'random',  # 使用随机划分数据集
    '--split_sizes', '0.8', '0.1', '0.1',  # 训练集80%，验证集10%，测试集10%
    '--gpu', '0',  # 使用GPU训练
    '--metric', 'rmse',  # 主要评价指标为 RMSE
    '--extra_metrics', 'r2',  # 额外的评价指标为 R²
    '--seed', '3407',  # 设置随机种子为 3407
    '--ensemble_size', '1',  # 设置保存的模型数量
    '--num_workers', '1',  # 设置数据加载的线程数，确保顺序一致，减少随机性
    '--no_cache_mol',  # 禁用缓存来确保数据加载的一致性
])

#print(f"特征文件的维度: {features_data.shape[1]}")

# 使用 cross_validate 函数进行训练
cross_validate(
    args=train_args,
    train_func=run_training  # 指定 run_training 作为训练函数
)
