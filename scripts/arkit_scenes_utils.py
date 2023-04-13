
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
    

def transform_3dod(video_id, image_id):


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_data(n_scenes=10)

