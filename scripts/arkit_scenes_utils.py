
import sys 
import os
sys.path.append(os.path.abspath("../ARKitScenes/"))
from download_data import download_data
import pandas as pd
import argparse
import numpy as np
import json
import cv2

sys.path.append(os.path.abspath("/home/john/homework/slam/CMU_16833_Semantic_Scene_Recognition/ARKitScenes/depth_upsampling/"))
from dataset import ARKitScenesDataset

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

def load_json(js_path):
    with open(js_path, "r") as f:
        json_data = json.load(f)
    return json_data

def compute_box_3d(scale, transform, rotation):
    scales = [i / 2 for i in scale]
    l, h, w = scales
    center = np.reshape(transform, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(np.transpose(rotation),
                        np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    bbox3d_raw = np.transpose(corners_3d)
    return bbox3d_raw

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: timestamp
        Rt: transformation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

def transform_3dod(scene_annotations, image_file, traj_line, intrinsics_file, sky_direction):
    """
    scene_annotations: -> File that contains scene annotations in scene, wrt venue frame
    image file: -> file of RGB image
    traj_line: -> traj in format: timestap (1,), rotation, axis-angle (2-4), translation, m (5-7)
    intrinsics: -> width height focal_length_x focal_length_y principal_point_x principal_point_y
    """

    # Load Annotations    
    annotation = load_json(scene_annotations)

    # Convert Camera Pose into rotation and transform
    _, cam_transformation_matrix = TrajStringToMatrix(traj_line)

    # Iterate through scene annotations, and transform to get 3D bbox data
    transformation_matrix = np.zeros((4,4))
    transformation_matrix[3,3] = 1
    bbox_list = []
    for label_info in annotation["data"]:
        rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
        transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)

        # Update transform to camera frame
        transformation_matrix[:3,:3] = rotation
        transformation_matrix[:3,3] = transform
        updated_transform = transformation_matrix @ cam_transformation_matrix
        # Decompose updated transform
        updated_rotation = updated_transform[:3,:3]
        updated_translate = updated_transform[:3,3]

        scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        box3d = compute_box_3d(scale.reshape(3).tolist(), updated_translate, updated_rotation)
        bbox_list.append(box3d)
    bbox_list = np.asarray(bbox_list)

    # Project bboxes into image for verification
    image = ARKitScenesDataset.load_image(image_file, (192, 256), False, sky_direction)
    transpose_image = image.transpose((1, 2, 0))
    cv2.imshow("Test", transpose_image.astype(int))
    intrinsics = st2_camera_intrinsics(intrinsics_file)
    proj_points, _ = cv2.projectPoints(bbox_list.reshape(-1,3), np.eye(3), np.array([[0.,0.,0.]]), intrinsics, None)
    for pt in proj_points:
        image = cv2.circle(image, tuple(pt.astype(int)[0]), 0, (255, 0, 0), -1)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    scene_ids = get_data(n_scenes=10)

    for scene_id in scene_ids:
        # Get images in relevant directory
        scene_dir = f"../ARKitScenes/data/3dod/Training/{scene_id}/"
        bbox_annotations = scene_dir + f"{scene_id}_3dod_annotation.json"
        images = os.listdir(scene_dir + f"{scene_id}_frames/lowres_wide")
        intrinsics = os.listdir(scene_dir + f"{scene_id}_frames/lowres_wide_intrinsics")
        traj_file = scene_dir + f"{scene_id}_frames/lowres_wide.traj"
        traj = open(traj_file, "r").readlines()
        metadata = pd.read_csv("../ARKitScenes/data/3dod/metadata.csv")

        for i in range(len(traj)):
            transform_3dod(
                bbox_annotations, 
                os.path.join(scene_dir + f"{scene_id}_frames/lowres_wide", images[i]), 
                traj[i], 
                os.path.join(scene_dir + f"{scene_id}_frames/lowres_wide_intrinsics", intrinsics[i]), metadata.sky_direction[i])
