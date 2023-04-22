import argparse
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def precision_at_n(proposals: dict, n: int):
    true_positives = 0
    false_positives = 0
    for targets, proposal_set in proposals.items():
        target_number = int(targets.split('_')[0])
        # Compute precision and recall
        if proposal_set is None:
            continue
        for i in range(n):
            proposal_number = int(proposal_set[i]['file_name'].split('_')[0])
            if proposal_number == target_number:
                true_positives += 1
            else:
                false_positives += 1
    precision = true_positives / (true_positives + false_positives)

    return precision

def pr_curve(proposals: dict):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    precision_values = []
    recall_values = []
    score_match_pairs = []
    for targets, proposal_set in proposals.items():
        # Visualize precision-recall curve
        target_number = int(targets.split('_')[0])
        if proposal_set is None:
            continue
        for proposal in proposal_set:
            proposal_number = int(proposal['file_name'].split('_')[0])
            score_match_pairs.append(np.array([proposal['score'], proposal_number == target_number]))

    score_match_pairs = sorted(score_match_pairs, key=lambda x: x[0], reverse=True)

    thresholds = np.linspace(0, 1, 101)
    for threshold in tqdm(thresholds):
        # Compute true positives, false positives, and false negatives for the given threshold
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for score, is_match in score_match_pairs:
            if score >= threshold:
                if is_match:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if is_match:
                    false_negatives += 1

        # Compute precision and recall for the given threshold
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        # Append precision and recall values to lists
        precision_values.append(precision)
        recall_values.append(recall)

    plt.plot(recall_values, precision_values)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

    return np.sum(precision_values) / len(precision_values)

def evaluate_proposals(proposals: dict):
    map = pr_curve(filtered_proposals)
    print("Mean Average Precision: ", map)

    top_n_values = [1, 3, 5, 10, 20, 50, 100]
    precision = []
    for n in top_n_values:
        precision.append(precision_at_n(filtered_proposals, n))

    for i in range(len(top_n_values)):
        print(f"Precision at n={top_n_values[i]}: {precision[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filtered_proposals", type=str, default="filtered_proposals.yaml"
    )
    args = parser.parse_args()

    f = open("scripts/" + args.filtered_proposals)
    data = yaml.load(f, Loader=yaml.SafeLoader)

    filtered_proposals = {}
    for target, proposals in data.items():
        filtered_proposals[target] = proposals
    
    evaluate_proposals(filtered_proposals)
