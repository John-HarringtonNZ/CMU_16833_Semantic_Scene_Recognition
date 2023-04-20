import sys 
import os
import glob
import shutil
import random 

def copy_videos(source, destination_memory, destination_target, data_set_percent_size = float(0.07)):

    # Get the subdirectory names using list comprehension
    video_ids = [d for d in os.listdir(source)]

    for video_id in video_ids:
        scene_dir = os.path.join(source, video_id)
        frame_folder = scene_dir + "/" + f"{video_id}_frames/lowres_wide"
        

        files = [f for f in os.listdir(frame_folder) if f.endswith('.png')]

        random.shuffle(files)
    
        split_num = int(len(files)*data_set_percent_size)
        files_memory = files[:split_num]
        files_target = files[split_num:]

        for file in files_memory:      
            src_path = os.path.join(frame_folder, file)
            dst_path = destination_memory
            shutil.copy(src_path, dst_path)
                    
        for file in files_target:      
            src_path = os.path.join(frame_folder, file)
            dst_path = destination_target
            shutil.copy(src_path, dst_path)


if __name__ == "__main__":

    source = "../ARKitScenes/data/3dod/Training/"
    destination_memory = "../ARKitScenes/memory"
    destination_target = "../ARKitScenes/target"
    os.makedirs(os.path.dirname(destination_memory), exist_ok=True)
    os.makedirs(os.path.dirname(destination_target), exist_ok=True)

    copy_videos(source, destination_memory, destination_target, data_set_percent_size = float(0.07))

    source1 = "../ARKitScenes/data/3dod/Training/40753679/40753679_frames/lowres_wide"
    source2 = "../ARKitScenes/data/3dod/Training/41254925/41254925_frames/lowres_wide"

    original_count1 = 0
    for root_dir, cur_dir, files in os.walk(source1):
        original_count1 += len(files)
    print('file count:', original_count1)
    
    original_count2 = 0
    for root_dir, cur_dir, files in os.walk(source2):
        original_count2 += len(files)
    print('file count:', original_count2)

    print("original count:", original_count1+original_count2)

    memory_count = 0
    for root_dir, cur_dir, files in os.walk(destination_memory):
        memory_count += len(files)
    print('memory count:', memory_count)

    target_count = 0
    for root_dir, cur_dir, files in os.walk(destination_target):
        target_count += len(files)
    print('target count:', target_count)