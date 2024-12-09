import pandas as pd
from chemprop.train import make_predictions
from chemprop.args import PredictArgs

if __name__ == '__main__':
    # 1. 读取预测数据
    prediction_data_path = 'assayyuce.csv'
    data = pd.read_csv(prediction_data_path)

    # 2. 定义预测参数，使用保存的模型目录
    predict_args = PredictArgs().parse_args([
        '--test_path', prediction_data_path,  # 测试数据路径
        '--checkpoint_dir', 'chemprop_checkpoints/PXR/fold_0',  # 使用保存多个集成模型的路径
        '--preds_path', 'assay_predictions.csv',  # 预测结果保存路径
        '--smiles_columns', 'smiles',  # SMILES 列名
        '--gpu', '0',  # 使用 GPU 进行预测
    ])

    # 3. 进行预测
    make_predictions(args=predict_args)

    # 4. 打印完成后的消息
    print("预测完成，结果保存在:", predict_args.preds_path)
