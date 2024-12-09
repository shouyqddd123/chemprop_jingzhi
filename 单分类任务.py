import pandas as pd
from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs

# 使用 main 函数保护代码
if __name__ == '__main__':
    # 2. 读取数据
    data_path = 'CAR.csv'
    data = pd.read_csv(data_path)

    # 3. 打印原始数据（可选，确保数据读取正常）
    print("原始数据形状:", data.shape)
    print(data.head())

    # 4. 定义训练参数
    train_args = TrainArgs().parse_args([
        '--data_path', data_path,  # 数据文件路径
        '--dataset_type', 'classification',  # 任务类型为分类
        '--save_dir', 'chemprop_checkpoints/CYP3a4',  # 模型保存路径
        '--target_columns', 'CAR',  # 要预测的标签列名
        '--smiles_columns', 'smiles',  # SMILES 列名
        '--epochs', '50',  # 训练轮数
        '--batch_size', '32',  # 每批样本数量
        '--hidden_size', '600',  # GNN的输出维数
        '--depth', '6',  # GNN的层数
        '--ffn_hidden_size', '300',  # 前馈网络的输出维数
        '--ffn_num_layers', '3',  # 前馈网络的层数
        '--dropout', '0.3',  # Dropout 率
        '--split_type', 'random',  # 使用随机划分数据集
        '--split_sizes', '0.8', '0.1', '0.1',  # 训练集80%，验证集10%，测试集10%
        '--gpu', '0',  # 使用 GPU 训练
        '--metric', 'auc',  # 主要评价指标为 AUC
        '--extra_metrics', 'accuracy', 'f1',  # 额外的评价指标为准确率和 F1 分数
        '--seed', '3407',  # 设置随机种子为 3407
        '--ensemble_size', '5',  # 设置保存的模型数量

    ])

    # 5. 使用 cross_validate 函数进行训练
    cross_validate(
        args=train_args,
        train_func=run_training  # 指定 run_training 作为训练函数
    )

    # 6. 完成训练后的消息（可选）
    print("模型训练完成并保存到:", train_args.save_dir)
