# Evaluation Instructions
- We recommend evaluating on one GPU. The code technically runs for multiple GPUs as well, but we have not implemented the final averaging of the evaluation metrics across GPUs.

## Classification
- For classification evaluation, simply adapt the command from [training instructions](training_instructions.md) by adding the `--eval_only` flag. 
- The command should look like this:
  ```
  python train_net.py \
  --eval_only \
  --snapshot_dir <path to the model checkpoint> \
  --dataset <dataset name> \
  <other required arguments>
  ```
- There is no need to specify the `--wandb` flag for evaluation. All the metrics will be printed to the console.

## Specific Commands
- For classification evaluation for models trained on CUB on the CUB test set (In-distribution eval), use the following command:
    ```
    python train_net.py \
    --eval_only \
    --data_path <base path to dataset>/CUB_200_2011 \
    --dataset cub \
    --image_sub_path_train images \
    --image_sub_path_test images \
    --train_split 1 \
    --eval_mode test \
    --image_size 518 \
    --augmentations_to_use cub_original \
    < model specific parameters >
    ```
- For classification evaluation for models trained on CUB on the Waterbirds test set (Out-of-distribution eval), make the following changes:
    ```
    python train_net.py \
    --eval_only \
    --image_sub_path_train waterbird_complete95_forest2water2 \
    --image_sub_path_test waterbird_complete95_forest2water2 \
    <other dataset specific arguments>
    < model specific parameters >
    ```
- To evaluate models trained on Waterbirds dataset, use the following commands:
  - For average accuracy on the test set:
      ```
      python train_net.py \
      --eval_only \
      --data_path <base path to dataset>/CUB_200_2011 \
      --dataset waterbirds \
      --image_sub_path_train waterbird_complete95_forest2water2 \
      --image_sub_path_test waterbird_complete95_forest2water2 \
      --train_split 1 \
      --eval_mode test \
      --image_size 518 \
      --augmentations_to_use cub_original \
      < model specific parameters >
      ```
  - For worst-group accuracy on test set (the worst group is the minimum of per-class accuracies):
      ```
      python train_net.py \
      --eval_mode worst_case \
      <other dataset specific arguments>
      < model specific parameters >
      ```
- To evaluate models trained on the Metashifts dataset, use the following command:
    ```
    python train_net.py \
    --eval_only \
    --data_path <base path to dataset>/MetaDatasetCatDog \
    --dataset metashifts \
    --image_sub_path_train train \
    --image_sub_path_test test \
    --train_split 1 \
    --eval_mode test \
    --image_size 224 \
    --augmentations_to_use cub_original \
    < model specific parameters >
    ```
    - In this case, the average accuracy is the micro-averaged accuracy across the two classes (cat and dog).
    - The worst-group accuracy is the minimum of the per-class accuracies.

- To evaluate models trained on the SIIM-ACR dataset, use the following commands:
  - For average AUROC:
      ```
      python train_net.py \
      --eval_only \
      --model_arch microsoft/rad-dino \
      --use_hf_transformers \
      --pretrained_start_weights \
      --data_path <base path to dataset>/siim-acr-pneumothorax-segmentation-data/versions/1 \
      --dataset siim_acr \
      --image_sub_path_train train_set \
      --image_sub_path_test full_test_set \
      --mask_sub_path all_masks \
      --image_size 518 \
      --augmentations_to_use siim_acr \
      < model specific parameters >
      ```
  - For robust AUROC:
      ```
      python train_net.py \
      --image_sub_path_test robust_test_set \
      <other dataset specific arguments>
      < model specific parameters >
      ```
    
## Evaluation with test-time interventions

### Low-confidence token removal
To evaluate the model with low-confidence token removal, use the following command:
```
python train_net.py \
--eval_only \
--part_logits_threshold_path <path to the part logits threshold file> \
<dataset specific arguments>
< model specific parameters >
```
The `part_logits_threshold_path` is the path to the file containing the threshold values for each part. The threshold values are used to remove the tokens with low-confidence part assignments. This in the form of a json file with keys as part indices and values as the threshold values.
These values are calculated using the training set.

To generate them, please use the following command:
```
python calculate_part_thresholds.py \
--model_path <path to the model checkpoint> \
--save_path <path to save the part logits threshold file> \
<dataset specific arguments>
< model specific parameters >
```


### Leave one part out eval
To evaluate the model with leave one part out evaluation, use the following command:
```
python leave_one_out_eval.py \
--eval_only \
<dataset specific arguments>
< model specific parameters >
```
- This script evaluates the model by leaving out one part at a time and measuring the performance of the model on the remaining parts. 
- It is also possible to evaluate the model by leaving out multiple parts at a time by specifying the `--intervention_fixed_part_id` argument. This argument specifies the part indices that should be left out during evaluation. The other parts are removed one at a time in this case.
- Additionally, the `--part_logits_threshold_path` argument can be used if you want to use both types of interventions together.

