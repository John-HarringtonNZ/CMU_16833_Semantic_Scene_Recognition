"""
Proposed Filter Types:

---Filter based on geometric matching
    Get coordinate frame, mIOU of 3D bounding boxes

---Filter based on semantic type matching
    Feasible instance counts

---Filter based on sizes of objects
    

---Filter based on color of objects

"""

import argparse
import json
from arkit_scenes_utils import *
from collections import Counter
import yaml

def get_scene_annotation(proposal_file):
    video_id = proposal_file.split("_")[0]
    annotation_file = f"../ARKitScenes/data/3dod/{video_id}/{video_id}_3dod_annotation.json"
    return load_json(annotation_file)

def get_scene_semantic_counts(annotation):
    """
    annotation -> json file, eg. "{...}_3dod_annotation.json"
    """
    labels = [label_info["label"] for label_info in annotation["data"]]
    return Counter(labels)

def identity_filter(target_file, proposals):
    """
    target_file -> string for file location
    proposals -> [Dict{'file_name':xx.png, 'score': 0.9}]
    """
    return proposals

def proposal_filter(target_file, base_proposals, filter_funcs):

    # Apply filter functions
    filtered_proposals = base_proposals
    for filter_func in filter_funcs:
        filtered_proposals = filter_func(target_file, filtered_proposals)

    return filtered_proposals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--proposals",type=str, default='../DBoW2/build/output.yaml'
    )
    parser.add_argument(
        "--memory-dir", type=str, default='ARKitScenes/memory'
    )
    parser.add_argument(
        "--target-dir", type=str, default='ARKitScenes/target'
    )
    args = parser.parse_args()

    f = open(args.proposals)
    data = yaml.load(f, Loader=yaml.SafeLoader)

    filtered_proposals = {}
    for target, proposals in data.items():
        filtered_proposals[target] = proposal_filter(target, proposals, [identity_filter])

    with open(f"filtered_proposals.yaml", 'w') as outfile:
        yaml.dump(data, outfile)