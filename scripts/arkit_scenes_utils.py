
import sys 
import os
sys.path.append(os.path.abspath("../ARKitScenes/"))
from download_data import download_data
import pandas as pd
import argparse


def get_data(n_scenes=1):

    # Look through 3dod_train_val_splits.csv
    df = pd.read_csv("../ARKitScenes/threedod/3dod_train_val_splits.csv")
    
    # Drop duplicate site visits
    df = df.drop_duplicates('visit_id')
    df = df.loc[df['visit_id'].isin(df.visit_id.dropna().unique())]
    video_ids_ = [str(id) for id in df.video_id.head(n_scenes).unique().tolist()]
    
    # Pick the first n_scenes items with unique visit_ids
    # Run the download script for all of them
    download_data('3dod',
                video_ids_,
                ["Training", ] * len(video_ids_),
                "../ARKitScenes/data/",
                False,
                None,
                False)
        
    return video_ids_

def transform_3dod(scene_annotations, image_file, traj_line, intrinsics_file):
    """
    scene_annotations: -> File that contains scene annotations in scene, wrt venue frame
    image file: -> file of RGB image
    traj_line: -> traj in format: timestap (1,), rotation, axis-angle (2-4), translation, m (5-7)
    intrinsics: -> width height focal_length_x focal_length_y principal_point_x principal_point_y
    """

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    scene_ids = get_data(n_scenes=10)

    for scene_id in scene_ids:
        # Get images in relevant directory
        scene_dir = "../ARKitScenes/data/3dod/Training/{scene_id}/"
        bbox_annotations = scene_dir + "{scene_dir}_3dod_annotation.json"
        images = os.listdir(scene_dir + "{scene_dir}_frames/lowres_wide")
        intrinsics = os.listdir(scene_dir + "{scene_dir}_frames/lowres_wide_intrinsics")
        traj_file = scene_dir + "{scene_dir}_frames/lowres_wide.traj"
        traj = open(traj_file, "r").readlines()

        for i in range(len(traj)):
            transform_3dod(bbox_annotations, images[i], traj[i], intrinsics[i])
