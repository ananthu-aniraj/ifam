import os
import argparse
import urllib.request
import torch
from huggingface_hub import HfApi
from models import FullTwoStageModelDoubleClassify, FullTwoStageModelDoubleClassifyHF

# Define the models configuration
MODELS_TO_UPLOAD = {
    "waterbirds-k4": {
        "class": "FullTwoStageModelDoubleClassify",
        "url": "https://github.com/ananthu-aniraj/ifam/releases/download/ifam-models/4_parts_waterbirds_snapshot_best.pt",
        "kwargs": {
            "model_arch": "vit_base_patch14_reg4_dinov2.lvd142m",
            "img_size": 518,
            "num_landmarks": 4,
            "num_classes": 2,
            "gumbel_softmax": True,
        },
        "description": "IFAM model trained on the Waterbirds dataset with 4 parts (K=4).",
        "dataset": "waterbirds",
        "num_parts": 4
    },
    "waterbirds-k8": {
        "class": "FullTwoStageModelDoubleClassify",
        "url": "https://github.com/ananthu-aniraj/ifam/releases/download/ifam-models/8_parts_waterbirds_snapshot_best.pt",
        "kwargs": {
            "model_arch": "vit_base_patch14_reg4_dinov2.lvd142m",
            "img_size": 518,
            "num_landmarks": 8,
            "num_classes": 2,
            "gumbel_softmax": True,
        },
        "description": "IFAM model trained on the Waterbirds dataset with 8 parts (K=8).",
        "dataset": "waterbirds",
        "num_parts": 8
    },
    "cub-k8": {
        "class": "FullTwoStageModelDoubleClassify",
        "url": "https://github.com/ananthu-aniraj/ifam/releases/download/ifam-models/8_parts_cub_snapshot_best.pt",
        "kwargs": {
            "model_arch": "vit_base_patch14_reg4_dinov2.lvd142m",
            "img_size": 518,
            "num_landmarks": 8,
            "num_classes": 200,
            "gumbel_softmax": True,
        },
        "description": "IFAM model trained on the CUB (Caltech-UCSD Birds-200-2011) dataset with 8 parts (K=8).",
        "dataset": "cub",
        "num_parts": 8
    },
    "metashift-k8": {
        "class": "FullTwoStageModelDoubleClassify",
        "url": "https://github.com/ananthu-aniraj/ifam/releases/download/ifam-models/8_parts_metashift_snapshot_best.pt",
        "kwargs": {
            "model_arch": "vit_base_patch14_reg4_dinov2.lvd142m",
            "img_size": 224,
            "num_landmarks": 8,
            "num_classes": 2,
            "gumbel_softmax": True,
        },
        "description": "IFAM model trained on the Metashifts dataset with 8 parts (K=8).",
        "dataset": "metashift",
        "num_parts": 8
    },
    "siim-acr-k8": {
        "class": "FullTwoStageModelDoubleClassifyHF",
        "url": "https://github.com/ananthu-aniraj/ifam/releases/download/ifam-models/8_parts_siim_acr_snapshot_best.pt",
        "kwargs": {
            "base_model_name_or_path": "microsoft/rad-dino",
            "num_landmarks": 8,
            "num_classes": 2,
            "gumbel_softmax": True,
        },
        "description": "IFAM model trained on the SIIM-ACR Pneumothorax Segmentation dataset with 8 parts (K=8).",
        "dataset": "siim_acr",
        "num_parts": 8
    }
}


def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download completed.")


def generate_model_card(local_dir, name, config, repo_id):
    desc = config["description"]
    class_name = config["class"]
    img_size = config["kwargs"].get("img_size", 518)
    num_parts = config["num_parts"]
    dataset = config["dataset"]

    readme_content = f"""---
license: apache-2.0
tags:
- pytorch
- image-classification
- robust-representation
- modular-ai
- dynamic-masking
- dinov2
- ifam
arxiv: 2506.08915
library_name: generic
---

# IFAM ({name}) Model Checkpoint

This is the official pre-trained checkpoint of the **IFAM (Iterative Focus and Attention Masking)** framework, proposed in the paper **"Two-stage Vision Transformers and Hard Masking offer Robust Object Representations"** (accepted as an oral presentation at ICPR 2026).

- **Paper:** [Two-stage Vision Transformers and Hard Masking offer Robust Object Representations](https://arxiv.org/abs/2506.08915)
- **Repository:** [GitHub - ananthu-aniraj/ifam](https://github.com/ananthu-aniraj/ifam)

## Model Description
{desc}

The IFAM framework is a two-stage approach:
1. **Stage 1 (Selector):** Processes the full image to discover object parts and identify task-relevant regions.
2. **Stage 2 (Predictor):** Restricts its receptive field to the selected regions using input attention masking, preventing spurious background details from affecting the classification.

## Usage

To use this model, ensure you have the `ifam` repository cloned and the dependencies installed. You can load this model directly from the Hugging Face Hub:

```python
import torch
import sys
# Append the path to the ifam repository
sys.path.append("/path/to/ifam")

from models import {class_name}

# Load the model directly from Hugging Face Hub
model = {class_name}.from_pretrained("{repo_id}")
model.eval()

# Example: Run inference
# x = torch.rand(1, 3, {img_size}, {img_size})
# with torch.no_grad():
#     outputs = model(x)
```

## Citation
If you use this model in your research, please cite:
```bibtex
@inproceedings{{aniraj2026ifam,
  title={{Two-stage Vision Transformers and Hard Masking offer Robust Object Representations}},
  author={{Aniraj, Ananthu and Dantas, Cassio F. and Ienco, Dino and Marcos, Diego}},
  booktitle={{International Conference on Pattern Recognition (ICPR)}},
  year={{2026}}
}}
```
"""
    with open(os.path.join(local_dir, "README.md"), "w") as f:
        f.write(readme_content)


def main():
    parser = argparse.ArgumentParser(description="Upload IFAM checkpoints to Hugging Face Model Hub")
    parser.add_argument("--username", type=str, default="ananthu-aniraj",
                        help="Your Hugging Face username or organization")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (write permission required)")
    parser.add_argument("--models", type=str, nargs="+", default=list(MODELS_TO_UPLOAD.keys()),
                        choices=list(MODELS_TO_UPLOAD.keys()), help="List of model keys to upload")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory to download weights")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "[WARNING] No Hugging Face token provided. Pushing to Hub might fail if you are not logged in via `hf auth login` / `huggingface-cli login`.")

    for name in args.models:
        config = MODELS_TO_UPLOAD[name]
        print(f"\n======================================")
        print(f"Processing model: {name}")
        print(f"======================================")

        # 1. Download checkpoint
        pt_filename = os.path.basename(config["url"])
        local_pt_path = os.path.join(args.cache_dir, pt_filename)
        if not os.path.exists(local_pt_path):
            download_file(config["url"], local_pt_path)
        else:
            print(f"Using cached checkpoint at {local_pt_path}")

        # 2. Instantiate correct model class
        print(f"Instantiating model class: {config['class']}")
        if config["class"] == "FullTwoStageModelDoubleClassify":
            model = FullTwoStageModelDoubleClassify(**config["kwargs"])
        elif config["class"] == "FullTwoStageModelDoubleClassifyHF":
            model = FullTwoStageModelDoubleClassifyHF(**config["kwargs"])
        else:
            raise ValueError(f"Unknown model class: {config['class']}")

        # 3. Load checkpoint state dict
        print("Loading weights from checkpoint...")
        ckpt = torch.load(local_pt_path, map_location="cpu")
        state_dict = ckpt["model_state"]

        # Strip any DDP prefixes if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")

        # 4. Save model locally in HF format
        local_hf_dir = os.path.join("./hf_models", f"ifam-{name}")
        os.makedirs(local_hf_dir, exist_ok=True)
        print(f"Saving HF formatted model to {local_hf_dir}...")
        model.save_pretrained(local_hf_dir)

        # 5. Generate README model card
        repo_id = f"{args.username}/ifam-{name}"
        generate_model_card(local_hf_dir, name, config, repo_id)
        print("Generated README.md (model card).")

        # 6. Push to HF Hub
        print(f"Pushing to HF Hub repo: {repo_id}...")
        try:
            from huggingface_hub import create_repo, upload_folder
            create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
            upload_folder(
                folder_path=local_hf_dir,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token
            )
            print(f"[SUCCESS] Model '{name}' uploaded successfully to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[ERROR] Failed to upload '{name}': {e}")
            print("Please ensure your token is valid and has write permissions.")


if __name__ == "__main__":
    main()
