
import sys 
import os
sys.path.append(os.path.abspath("../ARKitScenes/"))
from download_data import download_data
import pandas as pd
import argparse
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import glob

sys.path.append(os.path.abspath("../ARKitScenes/depth_upsampling/"))
from dataset import ARKitScenesDataset

sys.path.append(os.path.abspath("../ARKitScenes/threedod/benchmark_scripts/"))
from rectify_im import decide_pose, rotate_pose


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
    print(angle_axis)
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

def bboxes(annotation):
    bbox_list = []
    for label_info in annotation["data"]:
        rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
        transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
        bbox_list.append(box3d)
    bbox_list = np.asarray(bbox_list)
    return bbox_list

def transform_3dod(scene_annotations, image_file, traj_line, intrinsics_file, sky_direction):
    """
    scene_annotations: -> File that contains scene annotations in scene, wrt venue frame
    image file: -> file of RGB image
    traj_line: -> traj in format: timestap (1,), rotation, axis-angle (2-4), translation, m (5-7)
    intrinsics: -> width height focal_length_x focal_length_y principal_point_x principal_point_y
    """

    # Load Annotations and intrinsics
    annotation = load_json(scene_annotations)
    intrinsics = st2_camera_intrinsics(intrinsics_file)

    # Iterate through scene annotations, and transform to get 3D bbox data
    transformation_matrix = np.zeros((4,4))
    transformation_matrix[3,3] = 1

    # Project bboxes into image for verification
    print("Image: ")
    print(image_file)
    print("Sky direction", sky_direction)
    image = ARKitScenesDataset.load_image(image_file, (192, 256), False, "Up")
    image = image.transpose((1, 2, 0))

    bboxes3d = bboxes(annotation)

    centers = []
    for label_info in annotation["data"]:
        centers.append(np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3))
    centers = bboxes3d.reshape(-1,3)

    # Decompose updated transform
    _, cam_transformation_matrix = TrajStringToMatrix(traj_line) # Should be venue to camera
    cam_transformation_matrix = np.linalg.inv(cam_transformation_matrix)
    cam_rotation = cam_transformation_matrix[:3,:3]
    cam_translate = cam_transformation_matrix[:3,3]

    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    pts = centers.reshape(-1,3)
    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]
    
    # Creating plot
    ax.scatter3D(x, y, z, color = "green")
    ax.scatter3D(cam_translate[0], cam_translate[1], cam_translate[2], color='red')

    # Transform points in 3D
    transformed_pts = np.dot(cam_transformation_matrix, np.hstack([pts, np.ones((pts.shape[0],1))]).T).T
    transformed_x = transformed_pts[:,0]
    transformed_y = transformed_pts[:,1]
    transformed_z = transformed_pts[:,2]
    
    # Creating plot
    ax.scatter3D(transformed_x, transformed_y, transformed_z, color = "blue")

    plt.title("simple 3D scatter plot")

    # Strange result with NP advanced indexing within cv2 projectPoints
    filtered_pts = transformed_pts[np.where(transformed_pts[:,2] >=0)]

    # show plot
    proj_points, _ = cv2.projectPoints(filtered_pts[:,:3].reshape(-1,3).astype('float64'), np.eye(3), np.zeros((3,1)), intrinsics, None)
    for pt in proj_points:
        image = cv2.circle(image, tuple(pt.astype(int)[0]), 5, (255, 0, 0), -1)

    # Rotate for visualization
    direction2poseidx = {
        "Left": 1, "Right": 3, "Down": 2, "Up": 0
    }
    image = rotate_pose(image, direction2poseidx[sky_direction])

    cv2.imshow("Test Projection", image.astype('uint8'))
    cv2.waitKey()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    video_ids = get_data(n_scenes=10)

    for video_id in video_ids:
        scene_dir = f"../ARKitScenes/data/3dod/Training/{video_id}/"
        frame_folder = scene_dir + f"{video_id}_frames/lowres_wide"
        depth_images = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
        frame_ids = [os.path.basename(x) for x in depth_images]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        video_id = frame_folder.split('/')[-3]
        frame_ids = [float(x) for x in frame_ids]
        frame_ids.sort()

        # Get images in relevant directory
        bbox_annotations = scene_dir + f"{video_id}_3dod_annotation.json"
        images = sorted(os.listdir(scene_dir + f"{video_id}_frames/lowres_wide"))
        intrinsics = sorted(os.listdir(scene_dir + f"{video_id}_frames/lowres_wide_intrinsics"))
        traj_file = scene_dir + f"{video_id}_frames/lowres_wide.traj"
        traj = open(traj_file, "r").readlines()
        metadata = pd.read_csv("../ARKitScenes/data/3dod/metadata.csv")
        # Get valid sky_direction
        sky_direction = metadata.loc[metadata['video_id'] == int(video_id), 'sky_direction'].iloc[0]

        for i in range(len(traj)):

            intrinsics = scene_dir + f"{video_id}_frames/lowres_wide_intrinsics/" + video_id + "_" + "{:.3f}".format(frame_ids[i]) + ".pincam"
            image = scene_dir + f"{video_id}_frames/lowres_wide/" + video_id + "_" + "{:.3f}".format(frame_ids[i]) + ".png"

            transform_3dod(
                bbox_annotations, 
                image, 
                traj[i], 
                intrinsics, sky_direction)
