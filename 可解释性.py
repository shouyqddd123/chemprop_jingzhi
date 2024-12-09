import optuna
from chemprop.train import cross_validate, run_training, make_predictions
from chemprop.args import TrainArgs, PredictArgs
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import random
import torch
import os


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


# 使用特征置换评估特征重要性
def permutation_feature_importance(features_data, predict_data, predict_args, baseline_rmse, n_repeats=10):
    importances = {}

    # 遍历所有的特征列
    for column in features_data.columns:
        original_values = features_data[column].copy()
        rmses = []

        for _ in range(n_repeats):
            # 打乱特征列
            shuffled_values = original_values.sample(frac=1, random_state=3407).reset_index(drop=True)
            features_data[column] = shuffled_values

            # 将打乱后的特征数据保存回文件中，以便重新预测
            shuffled_features_path = 'shuffled_features.csv'
            features_data.to_csv(shuffled_features_path, index=False)

            # 创建新的预测参数，使用打乱后的特征文件
            shuffled_predict_args = PredictArgs().parse_args([
                '--test_path', 'yuce.csv',
                '--preds_path', 'shuffled_predictions.csv',
                '--features_path', shuffled_features_path,
                '--checkpoint_path', 'chemprop_checkpoints/fold_0/model_0/model.pt',
                '--gpu', '0',
            ])

            # 重新进行预测
            make_predictions(args=shuffled_predict_args)

            # 读取预测结果
            shuffled_predictions_df = pd.read_csv('shuffled_predictions.csv')
            print(shuffled_predictions_df.head())  # 打印出预测结果的前几行以确定列名

            # 自动检测预测列名
            possible_columns = shuffled_predictions_df.columns.difference(['smiles'])
            if len(possible_columns) == 1:
                shuffled_predictions = shuffled_predictions_df[possible_columns[0]]
            else:
                raise ValueError("预测结果中没有找到合适的列名，请检查文件内容")

            # 计算新的 RMSE
            rmse = np.sqrt(mean_squared_error(predict_data['target'], shuffled_predictions))
            rmses.append(rmse)

        # 计算特征置换后的重要性
        importances[column] = np.mean(rmses) - baseline_rmse

        # 恢复原始特征列
        features_data[column] = original_values

    return importances


# 主函数
if __name__ == "__main__":
    # 设置随机种子
    set_seed(3407)

    # 读取特征文件
    features_data = pd.read_csv('BCF_with_descriptors.csv')
    print(f"特征文件的形状: {features_data.shape}")
    print(features_data.head())

    # 定义训练参数
    train_args = TrainArgs().parse_args([
        '--data_path', 'BCF.csv',
        '--features_path', 'BCF_with_descriptors.csv',
        '--dataset_type', 'regression',
        '--save_dir', 'chemprop_checkpoints',
        '--target_columns', 'target',
        '--epochs', '50',
        '--batch_size', '32',
        '--hidden_size', '1007',
        '--depth', '2',
        '--ffn_hidden_size', '151',
        '--ffn_num_layers', '1',
        '--dropout', '0.30703855324064994',
        '--split_type', 'random',
        '--split_sizes', '0.8', '0.1', '0.1',
        '--gpu', '0',
        '--metric', 'rmse',
        '--extra_metrics', 'r2',
        '--seed', '3407',
        '--ensemble_size', '1',
        '--num_workers', '1',
        '--no_cache_mol',
    ])

    # 训练模型
    cross_validate(args=train_args, train_func=run_training)

    # 加载预测数据集
    predict_data = pd.read_csv('yuce.csv')
    features_path = 'yuce_额外特征.csv'
    predictions_save_path = 'predictions_with_results.csv'

    # 创建预测参数
    predict_args = PredictArgs().parse_args([
        '--test_path', 'yuce.csv',
        '--preds_path', predictions_save_path,
        '--features_path', features_path,
        '--checkpoint_path', 'chemprop_checkpoints/fold_0/model_0/model.pt',
        '--gpu', '0',
    ])

    # 运行预测
    make_predictions(args=predict_args)

    # 读取预测结果
    predictions = pd.read_csv(predictions_save_path)
    predicted_values = predictions.iloc[:, 1]

    # 将预测结果添加到原数据的第三列
    predict_data['predictions'] = predicted_values
    predict_data['target'] = pd.to_numeric(predict_data['target'], errors='coerce')
    predict_data['predictions'] = pd.to_numeric(predict_data['predictions'], errors='coerce')

    # 计算R²和RMSE
    r2 = r2_score(predict_data['target'], predict_data['predictions'])
    rmse = np.sqrt(mean_squared_error(predict_data['target'], predict_data['predictions']))
    print(f"R²: {r2}")
    print(f"RMSE: {rmse}")

    # 保存包含预测结果的CSV文件
    predict_data.to_csv('yuce_with_predictions.csv', index=False)
    print(predict_data)

    # 计算基准 RMSE
    baseline_rmse = rmse

    # 计算特征重要性
    feature_importances = permutation_feature_importance(features_data, predict_data, predict_args, baseline_rmse)
    print("Feature Importances:", feature_importances)
