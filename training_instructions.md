# Training Instructions

This document contains the instructions to train the models for the experiments in the paper.

The code has been designed to work with both single and multi-GPU training (including multi-node training) using PyTorch's Distributed Data Parallel (DDP) and the `torchrun` utility from [torchelastic](https://pytorch.org/docs/stable/elastic/run.html). It is also designed to auto-detect slurm environments and set the appropriate environment variables for multi-gpu training in DDP.

## Batch Size and Learning Rate
The `--batch_size` and `--lr` parameters can be used to set the batch size and learning rate for training.
The batch size is per GPU, so the total batch size will be `batch_size * num_gpus`.

In case you want to modify the batch size, please adjust the learning rate according to the [square root scaling rule](https://arxiv.org/abs/1404.5997). 
We use a base batch size of 16 for a starting learning rate of 1e-6 (backbone).
So, if you want to use a batch size of 32, you should use a learning rate of 1e-6 * sqrt(32/16) = 1e-6 * sqrt(2) = 1.414e-6. 
The scaling is not implemented in the training script, so you will have to manually adjust the learning rate. 

### Recommended Batch Sizes 
- For models trained on CUB/Waterbirds/SIIM-ACR, we recommend using a batch size of 16 (or higher per GPU - we trained with an overall batch size of 128 with 8 GPUs).
- For models trained on MetaShifts, we recommend using a batch size of 32 (or higher per GPU - we trained with an overall batch size of 128 with 4 GPUs). 

## Experiment Tracking
It is recommended to use [Weights and Biases](https://wandb.ai/site) for tracking the experiments. The `--wandb` flag can be used to enable this feature. Feel free to remove the `--wandb` flag if you don`t want to use it.
The command line parameters in the training script related to Weights and Biases are as follows:
- `--wandb`: Enable Weights and Biases logging
- `--wandb_project`: Name of the project in Weights and Biases
- `--group`: Name of the experiment group within the project 
- `--job_type`: Name of the type of job within the experiment group
- `--wandb_entity`: Name of the entity in Weights and Biases. This is usually the username or the team name in Weights and Biases. Please do not leave this empty if you are using Weights and Biases.
- `--wandb_mode`: Mode of logging in Weights and Biases. Use "offline" if the machine does not have internet access and "online" if it does. In case of "offline" mode, the logs can be uploaded to Weights and Biases later using the [wandb sync](https://docs.wandb.ai/ref/cli/wandb-sync) command.
- `--wandb_resume_id`: Resume a previous run in Weights and Biases. This is useful when you want to continue training from a previous run. Provide the run ID of the previous run to resume training. Use this in combination with the `--resume_training` flag to resume training from a previous checkpoint.
- `--log_interval`: The interval at which the logs are printed plus the gradients are logged to Weights and Biases. The default value is 10. Feel free to change this value as required.

## Training Command
The main training command for the experiments in the paper are provided below. Please read the [Dataset-specific Parameters](#dataset-specific-parameters) and [Model-specific Parameters](#model-specific-parameters) sections to adjust the parameters as required for your experiments.

For example, to reproduce the training of the model on the CUB dataset for K=8 foreground parts, you can use the following command to train the model on a single node with 4 GPUs:
```
torchrun \
--nnodes=1 \
--nproc_per_node=8 \
<base path to the code>/train_net.py \
--model_arch hf_hub:timm/vit_base_patch14_reg4_dinov2.lvd142m /
--pretrained_start_weights /
--data_path <base path to dataset>/CUB_200_2011 \
--batch_size 16 \
--wandb \
--epochs 90 \
--dataset cub \
--save_every_n_epochs <duration at which to save checkpoints> \
--num_workers 2 \
--image_sub_path_train images \
--image_sub_path_test images \
--train_split 1 \
--eval_mode test \
--wandb_entity <wandb username> \
--wandb_project <name of wandb project> \
--wandb_mode <online/offline> \
--job_type <wandb job type> \
--group <wandb group name> \
--snapshot_dir <path to save directory> \
--lr 2.828e-6 \
--optimizer_type adamw \
--scheduler_type cosine \
--scratch_lr_factor 1e4 \
--modulation_lr_factor 1e4 \
--finer_lr_factor 1e3 \
--drop_path_stage_2 0 \
--smoothing 0 \
--augmentations_to_use cub_original \ \
--num_parts 8 \
--weight_decay 0.05 \
--freeze_backbone \
--total_variation_loss 1.0 \
--concentration_loss 0.0 \
--enforced_presence_loss 2 \
--enforced_presence_loss_type enforced_presence \
--pixel_wise_entropy_loss 1.0 \
--gumbel_softmax \
--presence_loss_type original \
--grad_norm_clip 2.0
```

### Dataset-specific Parameters
- `--dataset`: The name of the dataset. For CUB, use `cub`. For Waterbirds, use `waterbirds`. For SIIM-ACR, use `siim_acr`. For MetaShifts, use `meta_shift`.
- `--data_path`: The path to the dataset. The folder structure should be as mentioned in the [README](README.md) file.
- `--image_sub_path_train`: The sub-path to the training images in the dataset. For instance, in the CUB dataset, the images are present in the `images` folder.
- `--image_sub_path_test`: The sub-path to the test images in the dataset. 
- `--train_split`: The split to use for training. Only applicable for CUB if you wish to train on a subset of the dataset. We always use the full dataset for training in the paper, so the default value is 1.
- `--eval_mode`: The mode for evaluation. Use `test` for evaluation on the test set and `val` for evaluation on the validation set. The default value is `test`. All the experiments in the paper are evaluated on the test set.
- `--augmentations_to_use`: The augmentations to use for training. The augmentations are defined in the [transform_utils](utils/data_utils/transform_utils.py) file. The default value is `cub_original` which uses standard augmentations from fine-grained classification literature. For SIIM-ACR, please use `siim_acr` as it contains grey-scale images. We also support a more sophisticated auto-augmentation policy, which can be used by setting the value to `timm`. This uses the auto-augment policy used by the [ConvNeXt paper](https://arxiv.org/abs/2201.03545).
- `--image_size`: The size of the input image. This value is set to 518 (default value for DinoV2 timm models) on CUB, Waterbirds, and SIIM-ACR. For MetaShifts, the value is set to 224. The value is set to 224 for MetaShifts as the images are of smaller resolution.
- `--mask_sub_path`: The sub-path to the masks in the dataset. This is only applicable for the SIIM-ACR dataset. The default value is `all_masks`.

#### Example Commands
- CUB dataset:
```
torchrun \
--nnodes=<number of machines> \
--nproc_per_node=<gpus per node> \
<base path to the code>/train_net.py \
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
- Waterbirds dataset:
```
torchrun \
--nnodes=<number of machines> \
--nproc_per_node=<gpus per node> \
<base path to the code>/train_net.py \
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
- SIIM-ACR dataset:
```
torchrun \
--nnodes=<number of machines> \
--nproc_per_node=<gpus per node> \
<base path to the code>/train_net.py \
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
- MetaShifts dataset:
```
torchrun \
--nnodes=<number of machines> \
--nproc_per_node=<gpus per node> \
<base path to the code>/train_net.py \
--data_path <base path to dataset>/MetaDatasetCatDog \
--dataset meta_shift \
--image_sub_path_train train \
--image_sub_path_test test \
--image_size 224 \
--augmentations_to_use cub_original \
< model specific parameters >
```
### Model-specific Parameters
- `--model_arch`: The architecture of the model. For the experiments in the paper, we use the [ViT-Base DinoV2](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) model. In theory, any model from the timm library supported by the [VisionTransformer class](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) can be used. Additionally, we also support all the torchvision and timm ResNet models and the timm ConvNeXt models. 
- `--use_hf_transformers`: Use the HuggingFace Transformers library for the model. This is used for loading the [RAD-DINO](https://huggingface.co/microsoft/rad-dino) model, in combination with the `--model_arch` and `--pretrained_start_weights` flag. This is used for the experiments in the paper for SIIM-ACR.
- `--num_parts`: The number of foreground parts to discover. Please adjust this value as required for the dataset. It is recommended to use the values specified in the paper. 
- `--pretrained_start_weights`: Use the pre-trained weights for the backbone. This requires an active internet connection. If you want to train from scratch, you can remove this flag.
- `--freeze_backbone`: Freeze the backbone weights. This will freeze all the layers in the ViT backbone (first stage), except for the layers we introduce for part discovery, the class, register tokens and position embeddings. This is used for the experiments in the paper.
- `--freeze_params_stage_1`: In combination with the `--freeze_backbone` flag, this can be used to freeze the entire model except for the part discovery layers. This can be used to freeze the class token, register token, and position embeddings in the first (part discovery stage) of the model.
- `--freeze_second_stage`: Freeze the second stage of the model. This is used to freeze the masked classification (second stage ViT) layers, except for the class token, register token, and position embeddings.
- `--freeze_params_stage_2`: In combination with the `--freeze_second_stage` flag, this can be used to freeze the entire second stage model except for the final classification layers. This can be used to freeze the class token, register token, and position embeddings in the second (masked classification) stage of the model.
- `--gumbel_softmax`: Use the Gumbel-Softmax trick (re-parametrization) on the part attention maps. This is used for the experiments in the paper.
- `--softmax_temperature`: The temperature for the Softmax operation. The default value is 1.0. We use this value for all the experiments in the paper.
- `--part_dropout`: The dropout probability for the part dropout (first stage). The default value is 0.3. We use this value for all the experiments in the paper.
- `--part_dropout_stage_2`: The dropout probability for the part dropout (second stage). The default value is 0.3. We use this value for all the experiments in the paper, except in the ablation study where we use multiple values.
- `--drop_path_stage_1`: The drop path probability for the ViT model (stage 1). The default value is 0.0. We use this value for all the experiments in the paper. 
- `--drop_path_stage_2`: The drop path probability for the ViT model (stage 2). The default value is 0.0. We use this value for all the experiments in the paper. 
- `--use_soft_masks`: Use soft masks for the second stage. This can be used to reproduce the ablation study in the paper.
- `--grad_norm_clip`: The maximum norm for the gradients. The default value is 2.0. We use this value for all the experiments in the paper.
- `--output_stride`: Only applicable if you use CNN models from timm. Not used for the experiments in the paper.

### Checkpointing and Logging Parameters 
- `--snapshot_dir`: The directory to save the checkpoints. Feel free to change this value as required.
- `--save_every_n_epochs`: The interval at which the checkpoints as well as (optionally) part assignment maps are saved. The default value is 16. Feel free to change this value as required. By default, the checkpoint with the best validation accuracy and the last checkpoint are saved. We use the model with the best validation accuracy for evaluation in the paper. 
- `--amap_saving_prob`: The probability of saving the part assignment maps. This is triggered on the first epoch, every save_every_n_epochs epoch and the last epoch. Set it to 0 to turn it off and 1 if you want to save it for every iteration. We recommend using a value of 0.05 during training and higher values such as 0.8 for evaluation. This can cause a significant slowdown during training as the maps are saved as images. 

### Optimizer and Scheduler Parameters
- `--optimizer_type`: The type of optimizer to use. The default value is `adamw`. We use this value for all the experiments in the paper.
- `--scheduler_type`: The type of scheduler to use. The default value is `cosine`. We use this value for all the experiments in the paper.
- `--scheduler_gamma`: The gamma value for the scheduler. Not used for the experiments in the paper, useful if you use step-based schedulers.
- `--scheduler_step_size`: The step size for the scheduler. Not used for the experiments in the paper, useful if you use step-based schedulers.
- `--lr`: The learning rate for training. Please refer to the [Batch Size and Learning Rate](#batch-size-and-learning-rate)  This particular value is used for the pre-trained layers (including tokens) in the first stage. 
- `--weight_decay`: The weight decay for the optimizer. The default value is `0.05`. We use this value for all the experiments in the paper. We have implemented the normalized weight decay formulation from the [AdamW paper](https://arxiv.org/abs/1711.05101) in the code.
- `--scratch_lr_factor`: The learning rate factor for the scratch layers. The default value is 1e4. We use this value for all the experiments in the paper.
- `--modulation_lr_factor`: The learning rate factor for the modulation layers. The default value is 1e4. We use this value for all the experiments in the paper.
- `--finer_lr_factor`: The learning rate factor for the finer layers. The default value is 1e3. We use this value for all the experiments in the paper.
- `--stage_two_lr_factor`: The learning rate factor for the pre-trained layers in the second stage. The default value is 1.0. We use this value for all the experiments in the paper.

### Loss Hyper-Parameters
The loss hyperparameters are already set to the values used in the paper. See the [training arguments script](argument_parser_train_net.py) for more details.
Feel free to modify these values if it is required for your experiments.

### Extra Notes
- If you wish to train on a single GPU, you can remove the `torchrun` command and the `--nnodes` and `--nproc_per_node` flags. Then run it as `python <base path to the code>/train_net.py <arguments>`.
- Please note that the code is written with the assumption that all the visible GPUs are to be used for training. If you want to use a subset of the GPUs, you will have to manually set the `CUDA_VISIBLE_DEVICES` environment variable before running the training script. This is automatically done in slurm and other job schedulers environments, so you don`t have to worry about it in those cases.
- The `--pretrained_start_weights` flag is used to load the pre-trained weights for the backbone. This requires an active internet connection.
-  The weights will be saved in the `~/.cache/torch/hub/checkpoints` directory which is automatically detected in our code, if already present.
- (OPTIONAL) If you do not have an active internet connection, it is also possible to separately run the `create_model()` function [here](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) to download the weights for the vit-base model. This will auto-detect the `~/.cache/torch/hub/checkpoints` directory and save the weights there.



