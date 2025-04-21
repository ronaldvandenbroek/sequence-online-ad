import os
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

def load_and_average_csvs(folder_path):
    data = []

    columns = [
        "y_prob", "y_true",
    ]
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            # print(f"Processing file: {file}")
            file_path = os.path.join(folder_path, file)
            dataset_name = file_path.split("_")[2]
            
            df = pd.read_csv(file_path, names=columns, header=None)
            df["dataset"] = dataset_name

            data.append(df)

    result_df = pd.concat(data, ignore_index=True)
    return result_df

# Example usage
folder_path = "results/dae/raw/"
result_df = load_and_average_csvs(folder_path)

def cal_best_PRF(y_true, probas_pred):
    '''
    计算在任何阈值下，最好的precision，recall。f1
    :param y_true:
    :param probas_pred:
    :return:
    '''
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, probas_pred)

    epsilon = 1e-10  # Small constant to avoid division by zero
    f1s=(2*precisions*recalls)/(precisions+recalls+epsilon)
    f1s[np.isnan(f1s)] = 0

    best_index=np.argmax(f1s)

    aupr = average_precision_score(y_true, probas_pred)

    # Alternative F1-scoring
    # Find best F1 index, ensuring recall is at least 50%
    alt_best_index = np.argmax(f1s * (recalls >= 0.5))  

    # Compute best binary F1-score using the threshold
    best_threshold = thresholds[alt_best_index]
    y_pred = (probas_pred >= best_threshold).astype(int)

    binary_f1 = f1_score(y_true, y_pred, average='binary')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')


    return precisions[best_index],recalls[best_index],f1s[best_index],aupr,binary_f1,micro_f1,macro_f1

def best_threshold_via_pr_curve(group):
    # dataset_name = group['dataset'].iloc[0]
    y_true = group['y_true'].values
    y_prob = group['y_prob'].values

    prefix_p, prefix_r, prefix_f1, _, prefix_binary_f1, prefix_micro_f1, prefix_macro_f1 = cal_best_PRF(y_true, y_prob)

    return pd.Series({
        'best_precision': prefix_p,
        'best_recall': prefix_r,
        'best_f1': prefix_f1,
        'best_binary_f1': prefix_binary_f1,
        'best_micro_f1': prefix_micro_f1,
        'best_macro_f1': prefix_macro_f1,
    })

# Apply per dataset
result_df_f1 = result_df.groupby('dataset').apply(best_threshold_via_pr_curve).reset_index()

result_df_f1[['prefix', 'suffix']] = result_df_f1['dataset'].str.split('-', n=1, expand=True)
result_df_f1 = result_df_f1.drop(columns=['dataset', 'suffix'])

result_df_f1_agg = result_df_f1.groupby('prefix').agg(
    best_precision=('best_precision', 'mean'),
    best_recall=('best_recall', 'mean'),
    best_f1=('best_f1', 'mean'),
    best_binary_f1=('best_binary_f1', 'mean'),
    best_micro_f1=('best_micro_f1', 'mean'),
    best_macro_f1=('best_macro_f1', 'mean')
).reset_index()

# Sort using this custom order
custom_order = [
    "huge", "large", "medium", "small", "wide",
    "hospitalbilling", "bpic13"
]
result_df_f1_agg['prefix'] = pd.Categorical(result_df_f1_agg['prefix'], categories=custom_order, ordered=True)
results_sorted = result_df_f1_agg.sort_values('prefix')

results_sorted.round(3).to_csv("results/dae/dae_results_summary.csv", index=False)