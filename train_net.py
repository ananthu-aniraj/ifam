import torch
from timeit import default_timer as timer

from argument_parser_train_net import parse_args
from utils.data_utils.transform_utils import load_transforms
from utils.training_utils.optimizer_params import build_optimizer, layer_group_matcher_pdisco_2_stage
from utils.training_utils.scheduler_params import build_scheduler
from utils.misc_utils import sync_bn_conversion, check_snapshot
from utils.training_utils.ddp_utils import multi_gpu_check, get_local_rank
from utils.wandb_params import get_train_loggers
from engine.distributed_trainer_two_stage import launch_pdisco_2_stage_trainer
from load_dataset import load_train_test_datasets
from load_model import load_model_2_stage
from load_losses import load_classification_loss, load_loss_hyper_params

torch.backends.cudnn.benchmark = True


def pdisco_train_eval():
    args = parse_args()

    train_loggers = get_train_loggers(args)

    # Create directory to save training checkpoints, otherwise load the existing checkpoint
    check_snapshot(args)

    # Get the transforms and load the dataset
    train_transforms, test_transforms = load_transforms(args)

    # Load the dataset
    dataset_train, dataset_test, num_cls = load_train_test_datasets(args, train_transforms, test_transforms)

    # Load the model
    model = load_model_2_stage(args, num_cls)

    # Check if there are multiple GPUs
    use_ddp = multi_gpu_check()
    # Convert BatchNorm to SyncBatchNorm if there is more than 1 GPU
    if use_ddp:
        model = sync_bn_conversion(model)

    local_rank = get_local_rank()
    model = model.to(local_rank, non_blocking=True)

    # Load the loss function
    loss_fn, mixup_fn = load_classification_loss(args, num_cls)

    # Load the loss hyperparameters
    loss_hyperparams, eq_affine_transform_params = load_loss_hyper_params(args)

    # Define the optimizer and scheduler
    param_groups, weight_decay = layer_group_matcher_pdisco_2_stage(args, model, dataset_train)
    optimizer = build_optimizer(args, param_groups, weight_decay)
    scheduler = build_scheduler(args, optimizer)
    # Start the timer
    start_time = timer()

    # Load averaging parameters
    averaging_params = {'type': args.averaging_type, 'decay': args.model_ema_decay,
                        'use_warmup': not args.no_model_ema_warmup,
                        'device': 'cpu' if args.model_ema_force_cpu else None}

    # Setup training and save the results
    launch_pdisco_2_stage_trainer(model=model,
                                  train_dataset=dataset_train,
                                  test_dataset=dataset_test,
                                  batch_size=args.batch_size,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  loss_fn=loss_fn,
                                  epochs=args.epochs,
                                  save_every=args.save_every_n_epochs,
                                  loggers=train_loggers,
                                  log_freq=args.log_interval,
                                  use_amp=args.use_amp,
                                  use_zero=args.use_zero,
                                  param_groups=param_groups,
                                  snapshot_path=args.snapshot_dir,
                                  grad_norm_clip=args.grad_norm_clip,
                                  num_workers=args.num_workers,
                                  mixup_fn=mixup_fn,
                                  seed=args.seed,
                                  eval_only=args.eval_only,
                                  loss_hyperparams=loss_hyperparams,
                                  eq_affine_transform_params=eq_affine_transform_params,
                                  use_ddp=use_ddp,
                                  sub_path_test=args.image_sub_path_test,
                                  dataset_name=args.dataset,
                                  amap_saving_prob=args.amap_saving_prob,
                                  grad_accumulation_steps=args.grad_accumulation_steps,
                                  averaging_params=averaging_params,
                                  )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    pdisco_train_eval()
