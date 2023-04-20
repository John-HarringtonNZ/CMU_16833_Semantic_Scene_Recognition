import argparse
import yaml

def identity_filter(target_file, proposals):
    return proposals

def evaluate_proposals(target, proposals, N=100):
    return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filtered_proposals",type=str, required=True
    )
    args = parser.parse_args()

    f = open(args.proposals)
    data = yaml.load(f, Loader=yaml.SafeLoader)

    filtered_proposals = {}
    for target, proposals in data.items():
        filtered_proposals[target] = evaluate_proposals(target, proposals, N=100)