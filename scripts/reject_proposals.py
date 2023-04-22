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
from arkit_scenes_utils import *
from collections import Counter
import yaml

def get_scene_annotation(proposal_file):
    video_id = proposal_file.split("_")[0]
    annotation_file = f"../ARKitScenes/data/3dod/Training/{video_id}/{video_id}_3dod_annotation.json"
    return load_json(annotation_file)

def get_scene_semantic_counts(annotation_data):
    """
    annotation -> json file, eg. "{...}_3dod_annotation.json"
    """
    labels = [label_info["label"] for label_info in annotation_data]
    return Counter(labels)

def get_traj_line(target_file):

    video_id = target_file.split("_")[0]
    truncated_frame_time = float(".".join(target_file.split("_")[1].split(".")[:2])[:-1])

    traj_file = f"../ARKitScenes/data/3dod/Training/{video_id}/{video_id}_frames/lowres_wide.traj"
    with open(traj_file) as fh:
        for line in fh:
            time = line.split(" ")[0]
            if abs(float(time) - truncated_frame_time) < 0.05:
                target_traj_line = line
                break

    return target_traj_line


def identity_filter(target_file, proposals, target_traj_line):
    """
    target_file -> string for file location
    proposals -> [Dict{'file_name':xx.png, 'score': 0.9}]
    """

    return proposals

def volume_comparison_filter(target_file, proposals, target_traj_line):
    """
    target_file -> string for file location
    proposals -> [Dict{'file_name':xx.png, 'score': 0.9}]
    """
    # Get annotation file
    target_annotation = get_scene_annotation(target_file)

    # Filter target by frustrum

    filtered_target_annotations, inds = filter_annotations_by_view_frustrum(target_annotation['data'], target_traj_line)

    # Get volumes of filtered_ target 
    filtered_target_volumes = get_filtered_target_volumes(filtered_target_annotations, target_traj_line)

    
    # Get volumes of proposal
    for proposal in proposals:
        proposal_annotation = get_scene_annotation(proposal['file_name'])
        # proposal_volumes = get_proposal_volumes(proposal_annotation)

        # TODO how to compare with target_volume and proposal volumes

    return filtered_proposals

def semantic_count_filter(target_file, proposals, target_traj_line):
    """
    target_file -> string for file location
    proposals -> [Dict{'file_name':xx.png, 'score': 0.9}]
    """

    # Get annotation file
    target_annotation = get_scene_annotation(target_file)

    # Filter target by frustrum
    filtered_target_annotations, inds = filter_annotations_by_view_frustrum(target_annotation['data'], target_traj_line)

    # Get Semantic scene count
    target_semantic_count = get_scene_semantic_counts(filtered_target_annotations)

    filtered_proposals = []
    for proposal in proposals:

        # Proposal annotation
        proposal_annotation = get_scene_annotation(proposal['file_name'])
        proposal_semantic_count = get_scene_semantic_counts(proposal_annotation['data'])

        # Check if semantic count is feasible 
        all_good = True
        for type in target_semantic_count.keys():
            
            if target_semantic_count[type] > proposal_semantic_count[type]:
                all_good = False
                break
        
        if all_good:
            filtered_proposals.append(proposal)

    return filtered_proposals

def proposal_filter(target_file, base_proposals, filter_funcs):

    # Don't know why we are getting bad cases like this
    if base_proposals is None: 
        return []

    target_traj_line = get_traj_line(target_file)
    # Apply filter functions
    filtered_proposals = base_proposals
    for filter_func in filter_funcs:
        filtered_proposals = filter_func(target_file, filtered_proposals, target_traj_line)
    print("pre/post filter: ", len(base_proposals), "->", len(filtered_proposals))

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

    filters = [
        identity_filter,
        semantic_count_filter
    ]

    filtered_proposals = {}
    for target, proposals in data.items():
        filtered_proposals[target] = proposal_filter(target, proposals, filters)

    with open(f"filtered_proposals.yaml", 'w') as outfile:
        yaml.dump(data, outfile)