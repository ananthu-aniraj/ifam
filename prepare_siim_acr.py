# Modified from https://github.com/khaledsaab/spatial_specificity/blob/main/src/data/cxr.py
import argparse
import os
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import cv2
import pickle
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare SIIM ACR dataset for training')
    parser.add_argument('--pkl_file_path', type=str,
                        help='Path to the pkl file containing the dictionary of image ids and tube presence')
    parser.add_argument('--root_dir', type=str,
                        help='Root directory where the dataset is stored')
    arguments = parser.parse_args()
    return arguments


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position: current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess_siim_acr(pkl_file_path, root_dir):
    cxr_tube_dict = read_pkl(pkl_file_path)

    # get segment annotations
    segment_df = pd.read_csv(os.path.join(root_dir, "train-rle.csv"))
    segment_df = segment_df.rename(
        columns={"ImageId": "id", " EncodedPixels": "encoded_pixels"}
    )
    # there are some image ids with multiple label rows, we'll just take the first
    segment_df = segment_df[~segment_df.id.duplicated(keep="first")]

    # get binary labels for pneumothorax, any row with a "-1" for encoded pixels is
    # considered a negative
    segment_df["target"] = (segment_df.encoded_pixels != " -1").astype(int)

    # get filepaths for all images in the "dicom-images-train" directory
    filepaths = sorted(glob.glob(os.path.join(root_dir, "dicom-images-train/*/*/*.dcm")))
    present_keys = []
    file_ids = [file_path.split("/")[-1].split(".dcm")[0] for file_path in filepaths]
    for key in cxr_tube_dict.keys():

        if any(x in key for x in file_ids):
            present_keys.append(key)

    filepath_df = pd.DataFrame(
        [
            {
                "filepath": filepath,
                "id": os.path.splitext(os.path.basename(filepath))[0],
            }
            for filepath in filepaths
        ]
    )

    df = segment_df
    # important to perform a left join here, because there are some images in the
    # directory without labels in `segment_df`
    df = df.merge(filepath_df, how="left", on="id")

    count_artefact_present = {"0": 0, "1": 0}
    artefact_present_positive_label_filepaths = []
    artefact_present_negative_label_filepaths = []
    count_artefact_absent = {"0": 0, "1": 0}
    artefact_absent_positive_label_filepaths = []
    artefact_absent_negative_label_filepaths = []
    # Check targets for filepaths in data
    for key in cxr_tube_dict.keys():
        if key in df['id'].values:
            if cxr_tube_dict[key] == 1:
                # artifact present
                label = df.loc[df['id'] == key, 'target'].values[0]
                count_artefact_present[str(label)] += 1
                if label == 1:
                    artefact_present_positive_label_filepaths.append(df.loc[df['id'] == key, 'filepath'].values[0])
                else:
                    artefact_present_negative_label_filepaths.append(df.loc[df['id'] == key, 'filepath'].values[0])
            else:
                # artifact absent
                label = df.loc[df['id'] == key, 'target'].values[0]
                count_artefact_absent[str(label)] += 1
                if label == 1:
                    artefact_absent_positive_label_filepaths.append(df.loc[df['id'] == key, 'filepath'].values[0])
                else:
                    artefact_absent_negative_label_filepaths.append(df.loc[df['id'] == key, 'filepath'].values[0])
    # Spurious correlated test set
    # Create directory for test set
    Path(os.path.join(root_dir, "non_robust_test_set/0")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root_dir, "non_robust_test_set/1")).mkdir(parents=True, exist_ok=True)

    for filepath in tqdm(artefact_present_positive_label_filepaths, desc="creating non_robust_test_set/1"):
        shutil.copy(filepath, os.path.join(root_dir, "non_robust_test_set/1"))
    for filepath in tqdm(artefact_absent_negative_label_filepaths, desc="creating non_robust_test_set/0"):
        shutil.copy(filepath, os.path.join(root_dir, "non_robust_test_set/0"))

    # Robust test set

    # Create directory
    Path(os.path.join(root_dir, "robust_test_set/0")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root_dir, "robust_test_set/1")).mkdir(parents=True, exist_ok=True)

    for filepath in tqdm(artefact_present_negative_label_filepaths, desc="creating robust_test_set/0"):
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(root_dir, "robust_test_set/0", filename))

    for filepath in tqdm(artefact_absent_positive_label_filepaths, desc="creating robust_test_set/1"):
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(root_dir, "robust_test_set/1", filename))

    # Full test set
    Path(os.path.join(root_dir, "full_test_set/0")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root_dir, "full_test_set/1")).mkdir(parents=True, exist_ok=True)
    for filepath in tqdm(artefact_present_negative_label_filepaths, desc="creating full_test_set/0 (robust)"):
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(root_dir, "full_test_set/0", filename))
    for filepath in tqdm(artefact_present_positive_label_filepaths, desc="creating full_test_set/1 (non-robust)"):
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(root_dir, "full_test_set/1", filename))
    for filepath in tqdm(artefact_absent_negative_label_filepaths, desc="creating full_test_set/0 (robust)"):
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(root_dir, "full_test_set/0", filename))
    for filepath in tqdm(artefact_absent_positive_label_filepaths, desc="creating full_test_set/1 (non-robust)"):
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(root_dir, "full_test_set/1", filename))

    # Train set
    # find all filepaths that are not in the test set
    train_filepaths = df.loc[~df['id'].isin(present_keys), 'filepath'].values
    # find labels for these filepaths
    train_labels = df.loc[~df['id'].isin(present_keys), 'target'].values

    # Create directory
    Path(os.path.join(root_dir, "train_set/0")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root_dir, "train_set/1")).mkdir(parents=True, exist_ok=True)

    for i, filepath in enumerate(tqdm(train_filepaths, desc="creating train_set")):
        filename = Path(filepath).name
        shutil.copy(filepath, os.path.join(root_dir, "train_set", str(train_labels[i]), filename))

    dest_path = os.path.join(root_dir, "all_masks")
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    print("Creating masks")
    for index, row in df.iterrows():
        if row["target"] == 1:
            mask = rle2mask(row["encoded_pixels"], 1024, 1024)
            cv2.imwrite(os.path.join(dest_path, row["id"] + ".png"), mask)
        else:
            mask = np.zeros((1024, 1024))
            cv2.imwrite(os.path.join(dest_path, row["id"] + ".png"), mask)


if __name__ == '__main__':
    args = parse_args()
    preprocess_siim_acr(args.pkl_file_path, args.root_dir)
