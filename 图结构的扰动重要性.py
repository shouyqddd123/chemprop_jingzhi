import pandas as pd
import numpy as np
import torch
import random
from rdkit import Chem
from sklearn.metrics import mean_squared_error
from chemprop.train import make_predictions
from chemprop.args import PredictArgs


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 删除节点并重新生成 SMILES
def delete_node_and_get_smiles(smiles, node_index):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RWMol(mol)  # 可编辑的分子对象
    mol.RemoveAtom(node_index)  # 删除节点
    new_smiles = Chem.MolToSmiles(mol)
    return new_smiles


# 运行基线预测，计算基准 RMSE
def baseline_prediction(predict_args, predict_data):
    make_predictions(args=predict_args)
    predictions = pd.read_csv(predict_args.preds_path)
    predicted_values = predictions.iloc[:, 1]
    predict_data['predictions'] = predicted_values
    baseline_rmse = np.sqrt(mean_squared_error(predict_data['target'], predicted_values))
    return baseline_rmse


# 计算图结构特征的重要性
def graph_structure_importance(predict_args, predict_data, baseline_rmse):
    importances = {}
    original_data = pd.read_csv(predict_args.test_path)

    for idx, row in original_data.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()

        for atom_idx in range(num_atoms):
            perturbed_smiles = delete_node_and_get_smiles(smiles, atom_idx)
            original_data.loc[idx, 'smiles'] = perturbed_smiles
            original_data.to_csv(predict_args.test_path, index=False)
            make_predictions(args=predict_args)
            perturbed_predictions = pd.read_csv(predict_args.preds_path).iloc[:, 1]
            rmse = np.sqrt(mean_squared_error(predict_data['target'], perturbed_predictions))
            importance = rmse - baseline_rmse
            importances[f'{smiles}_atom_{atom_idx}'] = importance
            original_data.loc[idx, 'smiles'] = smiles

    return importances


if __name__ == "__main__":
    # 为多进程提供支持
    torch.multiprocessing.freeze_support()

    set_seed(3407)
    predict_data = pd.read_csv('yuce.csv')
    features_path = 'yuce_额外特征.csv'
    predictions_save_path = 'predictions_with_results.csv'

    predict_args = PredictArgs().parse_args([
        '--test_path', 'yuce.csv',
        '--preds_path', predictions_save_path,
        '--features_path', features_path,
        '--checkpoint_path', 'chemprop_checkpoints/fold_0/model_0/model.pt',
        '--gpu', '0',
    ])

    baseline_rmse = baseline_prediction(predict_args, predict_data)
    print(f"Baseline RMSE: {baseline_rmse}")

    graph_importances = graph_structure_importance(predict_args, predict_data, baseline_rmse)
    print("Graph Structure Importances:", graph_importances)

    importances_df = pd.DataFrame(list(graph_importances.items()), columns=['Graph Structure', 'Importance'])
    importances_df.to_csv('graph_structure_importances.csv', index=False)
    print("Graph Structure Importances saved to 'graph_structure_importances.csv'")
