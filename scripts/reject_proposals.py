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
import random

CACHED_SCENE_ANNOTATIONS = {}

def get_scene_annotation(proposal_file):
    video_id = get_video_id(proposal_file)
    if video_id not in CACHED_SCENE_ANNOTATIONS:
        annotation_file = f"../ARKitScenes/data/3dod/Training/{video_id}/{video_id}_3dod_annotation.json"
        annotation = load_json(annotation_file)
        CACHED_SCENE_ANNOTATIONS[video_id] = annotation
    return CACHED_SCENE_ANNOTATIONS[video_id]

def get_scene_semantic_counts(annotation_data):
    """
    annotation -> json file, eg. "{...}_3dod_annotation.json"
    """
    labels = [label_info["label"] for label_info in annotation_data]
    return Counter(labels)

def get_traj_line(target_file):
    video_id = get_video_id(target_file)
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
    filtered_target_volumes = get_volumes(filtered_target_annotations, target_traj_line)     # DEPRECATED: filtered_target_volumes = get_scaled_volumes(filtered_target_annotations, target_traj_line)
    set_filtered_target_volumes = set(filtered_target_volumes)

    filtered_proposals = []
    for proposal in proposals:
        proposal_annotation = get_scene_annotation(proposal['file_name'])
        proposal_volumes = get_volumes(proposal_annotation['data'])
        set_proposal_volumes = set(proposal_volumes)
        # TODO how to compare with target_volume and proposal volumes
        if set_filtered_target_volumes.issubset(set_proposal_volumes):
            filtered_proposals.append(proposal)
        else:
            continue

    return filtered_proposals

def semantic_count_filter(target_file, proposals, target_traj_line, add_noise=True, noise_threshold=0.1):
    """
    target_file -> string for file location
    proposals -> [Dict{'file_name':xx.png, 'score': 0.9}]
    """

    # Get annotation file
    target_annotation = get_scene_annotation(target_file)

    # Filter target by frustrum
    filtered_target_annotations, inds = filter_annotations_by_view_frustrum(target_file, target_annotation['data'], target_traj_line)

    # Get Semantic scene count
    target_semantic_count = get_scene_semantic_counts(filtered_target_annotations)

    # Apply target dropout
    if add_noise:
        for item in target_semantic_count.keys():
            rand_n = random.random()
            to_remove = min(noise_threshold // rand_n, target_semantic_count[item])
            target_semantic_count[item] -= to_remove

    filtered_proposals = []
    for proposal in proposals:

        # Proposal annotation
        proposal_annotation = get_scene_annotation(proposal['file_name'])
        proposal_semantic_count = get_scene_semantic_counts(proposal_annotation['data'])

        # Apply proposal dropout
        if add_noise:
            for item in proposal_semantic_count.keys():
                rand_n = random.random()
                to_remove = min(noise_threshold // rand_n, proposal_semantic_count[item])
                proposal_semantic_count[item] -= to_remove

        # Check if semantic count is feasible 
        all_good = True
        for type in target_semantic_count.keys():
            
            if target_semantic_count[type] > proposal_semantic_count[type]:
                all_good = False
                break
        
        if all_good:
            filtered_proposals.append(proposal)

    return filtered_proposals

def bbox_center_alignment_filter(target_file, proposals, target_traj_line, noise_std=0.1, dropout_prob=0.1):
    target_annotation = get_scene_annotation(target_file)
    filtered_target_annotations, _ = filter_annotations_by_view_frustrum(target_file, target_annotation['data'], target_traj_line)
    if len(filtered_target_annotations) <= 1:
        # It doesn't make sense to run this filter if there are 1 or fewer bounding boxes in view
        return proposals
    target_bbox_info_unfiltered = bbox_labeled_centers(filtered_target_annotations)

    # Apply target dropout
    target_bbox_info = []
    for t in target_bbox_info_unfiltered:
        if np.random.rand() >= dropout_prob:
            target_bbox_info.append(t)
    target_labels = [t['label'] for t in target_bbox_info]
    target_centers = np.array([t['center'].flatten() for t in target_bbox_info])
    # Apply target geometric noise
    target_centers += np.random.normal(0, noise_std, size=target_centers.shape)
    filtered_proposals = []

    for proposal in proposals:
        proposal_annotation = get_scene_annotation(proposal['file_name'])
        proposal_bbox_info_unfiltered = bbox_labeled_centers(proposal_annotation['data'])

        # Apply proposal dropout
        proposal_bbox_info = []
        for p in proposal_bbox_info_unfiltered:
            if np.random.rand() >= dropout_prob:
                proposal_bbox_info.append(p)

        # Find the least frequent proposal label that is present in target_labels to use as the anchor
        proposal_labels = [p['label'] for p in proposal_bbox_info]
        proposal_centers = np.array([p['center'].flatten() for p in proposal_bbox_info])
        # Apply proposal geometric noise
        proposal_centers += np.random.normal(0, noise_std, size=proposal_centers.shape)
        proposal_label_counts = Counter(proposal_labels).most_common()
        proposal_label_counts.reverse()
        anchor_label = None
        for label,_ in proposal_label_counts:
            if label in target_labels:
                anchor_label = label
                break
        if anchor_label is None:
            continue
        anchor_idx = [i for i,label in enumerate(proposal_labels) if label == anchor_label]
        target_anchor_idx = target_labels.index(anchor_label)
        target_anchor = target_centers[target_anchor_idx]
        transformed_target_centers = target_centers - target_anchor
        min_total_error = np.inf
        is_good = True
        for ai in anchor_idx:
            proposal_anchor = proposal_centers[ai]
            transformed_proposal_centers = proposal_centers - proposal_anchor
            curr_total_error = 0.0
            for ti in range(len(transformed_target_centers)):
                if ti == target_anchor_idx:
                    continue
                # Get all indices of proposal bboxes with matching labels 
                proposal_idx = [i for i,label in enumerate(proposal_labels) if label == target_labels[ti]]
                # Compute min L2 distance from target point to any of the proposal points
                if not proposal_idx:
                    # This means we have target labels that do not exist in proposal labels (should be filtered out by semantic counts filter)
                    curr_total_error = np.inf
                    is_good = False
                    break
                min_error = np.min(np.linalg.norm(transformed_proposal_centers[proposal_idx] - transformed_target_centers[ti], axis=1))
                curr_total_error += min_error
            curr_total_error /= len(transformed_target_centers)
            min_total_error = min(min_total_error, curr_total_error)
        if min_total_error < max(6*noise_std,0.1) and is_good:
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
        "--output_file",type=str, default='filtered_proposals.yaml'
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
        volume_comparison_filter
        # semantic_count_filter,
        # bbox_center_alignment_filter
    ]

    filtered_proposals = {}
    for target, proposals in data.items():
        filtered_proposals[target] = proposal_filter(target, proposals, filters)

    with open(args.output_file, 'w') as outfile:
        yaml.dump(filtered_proposals, outfile)