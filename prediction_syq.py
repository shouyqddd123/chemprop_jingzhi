import pandas as pd
from chemprop.train import make_predictions
from chemprop.args import PredictArgs
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import random
import torch

# 设置随机种子确保一致性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # 设置随机种子
    set_seed(3407)

    # 加载预测数据集
    predict_data = pd.read_csv('yuce.csv')

    # 特征文件路径，与训练时使用的特征相同
    features_path = 'yuce_额外特征2.csv'

    # 预测结果保存路径
    predictions_save_path = 'predictions_with_results.csv'

    # 创建预测参数
    predict_args = PredictArgs().parse_args([
        '--test_path', 'yuce.csv',  # 输入预测数据的路径
        '--preds_path', predictions_save_path,  # 保存预测结果的路径
        '--features_path', features_path,  # 预测时使用的特征文件
        '--checkpoint_path', 'chemprop_checkpoints/fold_0/model_0/model.pt',  # 模型路径
        '--gpu', '0',  # 使用 GPU
    ])

    # 运行预测
    make_predictions(args=predict_args)

    # 读取预测结果，仅选择预测的目标列（通常是第二列）
    predictions = pd.read_csv(predictions_save_path)

    # 假设预测结果在第二列，如果不是，你需要根据实际情况调整
    predicted_values = predictions.iloc[:, 1]

    # 将预测结果添加到原数据的第三列
    predict_data['predictions'] = predicted_values

    # 确保预测列和目标列都是浮点型
    predict_data['target'] = pd.to_numeric(predict_data['target'], errors='coerce')
    predict_data['predictions'] = pd.to_numeric(predict_data['predictions'], errors='coerce')

    # 计算R²和RMSE
    r2 = r2_score(predict_data['target'], predict_data['predictions'])
    rmse = np.sqrt(mean_squared_error(predict_data['target'], predict_data['predictions']))

    # 输出R²和RMSE
    print(f"R²: {r2}")
    print(f"RMSE: {rmse}")

    # 保存包含预测结果的CSV文件
    predict_data.to_csv('yuce_with_predictions.csv', index=False)

    # 显示包含预测结果的前几行
    print(predict_data)
