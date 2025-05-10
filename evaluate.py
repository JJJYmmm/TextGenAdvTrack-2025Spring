import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import ParameterGrid

def eval_func(gt_name="ground_truth", path="", alpha = 0, beta = 0):
    """
    Args:
        gt_name (str): Name of the ground truth file (without extension)
        path (str): Path to directory containing result files
        
    Returns:
        dict: Dictionary with team/method names as keys and their metrics as values
    """
    ret = {}

    # Get all files in the directory
    teams = os.listdir(path)
    
    # Filter out ground truth and leaderboard files
    if gt_name + ".csv" in teams:
        teams.remove(gt_name + ".csv")
    if "LeaderBoard.xlsx" in teams:
        teams.remove("LeaderBoard.xlsx")

    # Read ground truth labels
    gts = pd.read_csv(
        os.path.join(path, gt_name + ".csv")
    )
    
    print(f"Found {len(teams)} team/method submissions to evaluate")
    
    for team in teams:
        try:
            # Read prediction results
            data = pd.read_excel(
                os.path.join(path, team),
                sheet_name="predictions",
                engine="openpyxl"
            )
            
            # Extract prediction values
            predictions = data["text_prediction"].values

            predictions = predictions * alpha + beta
            
            # Get ground truth labels
            ground_truth = gts["label"].values
            
            # Convert prediction probabilities to binary classification results (threshold 0.5)
            binary_pred = (predictions >= 0.5).astype(int)
            
            # Calculate AUC
            auc = roc_auc_score(ground_truth, predictions)
            
            # Calculate F1 score
            f1 = f1_score(ground_truth, binary_pred)
            
            # Calculate precision and recall
            precision = precision_score(ground_truth, binary_pred)
            recall = recall_score(ground_truth, binary_pred)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(ground_truth, binary_pred).ravel()
            
            # Calculate accuracy
            accuracy = accuracy_score(ground_truth, binary_pred)  # Convert to percentage
            
            # Read time information
            time_data = pd.read_excel(
                os.path.join(path, team),
                sheet_name="time",
                engine="openpyxl"
            )
            
            # Calculate average processing time per sample
            mean_time = time_data["Time"][0] / 100 / time_data["Data Volume"][0]
            
            # Store results
            ret[team.split(".")[0]] = {
                "auc": auc,
                "accuracy": accuracy,  # Acc.(%)
                "f1": f1,             # F1
                "fn": fn,             # FN
                "fp": fp,             # FP
                "precision": precision, # Prec
                "recall": recall,     # Rec
                "mean_time": mean_time
            }
            
            print(f"Evaluated {team}: AUC={auc:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
            
        except Exception as e:
            print(f"Error processing {team}: {str(e)}")
            continue

    return ret

def calculate_score(opts, alpha, beta):
    results = eval_func(gt_name=opts.gt_name, path=opts.submit_path, alpha=alpha, beta=beta)
    leaderboard_data = {
        "Team/Method": results.keys(),
        "AUC": [res["auc"] for res in results.values()],
        "Acc": [res["accuracy"] for res in results.values()],
        "F1": [res["f1"] for res in results.values()],
        "Avg Time (s)": [res["mean_time"] for res in results.values()]
    }
    leaderboard_df = pd.DataFrame(data=leaderboard_data)
    leaderboard_df["Weighted Score"] = (
        0.6 * leaderboard_df["AUC"] +
        0.3 * leaderboard_df["Acc"] +
        0.1 * leaderboard_df["F1"] / 100
    )

    return leaderboard_df["Weighted Score"].max()

def find_best_alpha_beta(opts):
    import random

    best_alpha = 1
    best_beta = 0
    num_trials = 1000
    best_score = calculate_score(opts, best_alpha, best_beta)

    for i in range(num_trials):
        alpha = random.uniform(0.0, 2.0)      # 随机选取 alpha 范围
        beta = random.uniform(-1.0, 1.0)      # 随机选取 beta 范围
        score = calculate_score(opts, alpha, beta)
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_beta = beta
            print(f"New best: alpha={alpha:.2f}, beta={beta:.2f}, score={score:.4f}")

    print("Best parameters found:")
    print(f"alpha = {best_alpha:.2f}, beta = {best_beta:.2f}, best_score = {best_score:.4f}")

    return best_alpha, best_beta, best_score

def find_best_alpha_beta_grid(opts):
    alpha_range = np.arange(0.0, 2.0, 0.1)     # 例如从 0 到 2，步长 0.1
    beta_range = np.arange(-1.0, 1.0, 0.1)     # 例如从 -1 到 1，步长 0.1

    best_alpha = 1
    best_beta = 0
    best_score = calculate_score(opts, best_alpha, best_beta)

    for alpha in alpha_range:
        for beta in beta_range:
            score = calculate_score(opts, alpha, beta)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_beta = beta
                print(f"New best: alpha={alpha:.2f}, beta={beta:.2f}, score={score:.4f}")

    print("Best parameters found:")
    print(f"alpha = {best_alpha:.2f}, beta = {best_beta:.2f}, best_score = {best_score:.4f}")

    return best_alpha, best_beta, best_score

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--submit-path",
        type=str,
        default="/mnt/data/jinxiaochuan/Project/UCAS/LLMDA/results",
        help="Path to directory containing submission files"
    )
    arg.add_argument(
        "--gt-name",
        type=str,
        default="/mnt/data/jinxiaochuan/Project/UCAS/LLMDA/data/test-gt",
        help="Name of ground truth file (without extension)"
    )
    opts = arg.parse_args()
    writer = pd.ExcelWriter(os.path.join(opts.submit_path, "LeaderBoard.xlsx"), engine="openpyxl")
        
    # best_alpha, best_beta, best_score = find_best_alpha_beta_grid(opts)
    
    # # Update the best alpha, beta, and weighted score if necessary

    # print(f"Best alpha: {best_alpha}")
    # print(f"Best beta: {best_beta}")
    # print(f"Best weighted score: {best_score}")

    # best_alpha = 1.60
    # best_beta = -0.87
    best_alpha = 1
    best_beta = 0

    results = eval_func(gt_name=opts.gt_name, path=opts.submit_path, alpha=best_alpha, beta=best_beta)
    leaderboard_data = {
        "Team/Method": results.keys(),
        "AUC": [res["auc"] for res in results.values()],
        "Acc": [res["accuracy"] for res in results.values()],
        "F1": [res["f1"] for res in results.values()],
        "Avg Time (s)": [res["mean_time"] for res in results.values()]
    }
    leaderboard_df = pd.DataFrame(data=leaderboard_data)
    leaderboard_df["Weighted Score"] = (
        0.6 * leaderboard_df["AUC"] +
        0.3 * leaderboard_df["Acc"] +
        0.1 * leaderboard_df["F1"] / 100
    )

    leaderboard_df.to_excel(writer, index=False)
    writer.close()