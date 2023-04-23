import argparse
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def precision_at_n(proposals: dict, n: int) -> float:
    # Intialize counter
    true_positives = 0
    false_positives = 0

    # Loop through all targets and proposals to accumulate tp and fp
    for targets, proposal_set in proposals.items():
        target_number = int(targets.split('_')[0].split('/')[-1])
        # Skip null targets
        if proposal_set is None:
            continue
        if len(proposal_set) < n:
            return -1
        for i in range(n):
            proposal_number = int(proposal_set[i]['file_name'].split('_')[0].split('/')[-1])
            if proposal_number == target_number:
                true_positives += 1
            else:
                false_positives += 1

    # Calculate precision
    precision = true_positives / (true_positives + false_positives)

    return precision

def get_score_match_pairs(proposals: dict) -> np.ndarray:
    # Intialize counter
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # List of precision and recall values
    precision_values = []
    recall_values = []

    # Stores all target-proposal score pairs
    score_match_pairs = []
    for targets, proposal_set in proposals.items():
        # Visualize precision-recall curve
        target_number = int(targets.split('_')[0].split('/')[-1])
        if proposal_set is None:
            continue
        for proposal in proposal_set:
            proposal_number = int(proposal['file_name'].split('_')[0].split('/')[-1])
            if proposal['score'] != 1:
                score_match_pairs.append(np.array([proposal_number == target_number, proposal['score']]))

    # Sort matches - could be used for optimization later if needed
    score_match_pairs = sorted(score_match_pairs, key=lambda x: x[1], reverse=True)
    score_match_pairs = np.array(score_match_pairs)
    return score_match_pairs

def pr_curve(proposals: dict) -> float:
    score_match_pairs = get_score_match_pairs(proposals)

    # Compute precision recall curve
    precision, recall, _ = precision_recall_curve(score_match_pairs[:,0], score_match_pairs[:,1])

    # Visualize the PR curve
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.show()

    plt.savefig("pr_curve")

    map = average_precision_score(score_match_pairs[:,0], score_match_pairs[:,1])

    return map

def evaluate_proposals(filtered_proposals: dict) -> None:
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
        "--filtered_proposals", type=str, default="../DBoW2/build/output.yaml"
    )
    args = parser.parse_args()

    f = open(args.filtered_proposals)
    data = yaml.load(f, Loader=yaml.SafeLoader)

    filtered_proposals = {}
    for target, proposals in data.items():
        filtered_proposals[target] = proposals
    proposals_name = args.filtered_proposals.split(".")[0]
    print(proposals_name)
    
    evaluate_proposals(filtered_proposals)
