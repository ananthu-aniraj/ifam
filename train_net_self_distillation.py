import torch
from timeit import default_timer as timer

from argument_parser_distillation_train import parse_args
from load_dataset import get_dataset
from load_model import load_model_distillation, load_model_pdisco
from load_losses import load_mse_loss, load_dino_loss
from utils.data_utils.transform_utils import load_transforms
from utils.misc_utils import sync_bn_conversion, check_snapshot
from utils.wandb_params import get_train_loggers
from utils.training_utils.optimizer_params import build_optimizer, layer_group_matcher_baseline
from utils.training_utils.scheduler_params import build_scheduler
from utils.training_utils.ddp_utils import multi_gpu_check
from engine.distributed_trainer_self_distillation import launch_distillation_trainer

torch.backends.cudnn.benchmark = True


def distillation_train_eval():
    args = parse_args()

    train_loggers = get_train_loggers(args)

    # Create directory to save training checkpoints, otherwise load the existing checkpoint
    check_snapshot(args)

    # Get the transforms and load the dataset
    args.use_albumentations = False
    train_transforms, test_transforms = load_transforms(args)
    dataset_train, dataset_test, num_cls = get_dataset(args, train_transforms, test_transforms)

    # Load the student model
    student_model, teacher_model = load_model_distillation(args, num_cls)

    if args.pdiscoformer_pretrained_path is not None:
        # Load Pdiscoformer model
        pdisco_model = load_model_pdisco(args, dataset_train.num_classes)
    else:
        pdisco_model = None

    # Check if there are multiple GPUs
    use_ddp = multi_gpu_check()
    # Convert BatchNorm to SyncBatchNorm if there is more than 1 GPU
    if use_ddp:
        student_model = sync_bn_conversion(student_model)
        teacher_model = sync_bn_conversion(teacher_model)
        if pdisco_model is not None:
            pdisco_model = sync_bn_conversion(pdisco_model)

    # Load the loss function
    if args.loss_type == 'mse':
        loss_fn, mixup_fn = load_mse_loss(args, reduction='mean')
    else:
        loss_fn, mixup_fn = load_dino_loss(output_dim=student_model.embed_dim)

    # Load the optimizer and scheduler
    param_groups = layer_group_matcher_baseline(args, student_model)
    optimizer = build_optimizer(args, param_groups, dataset_train)
    scheduler = build_scheduler(args, optimizer)

    # Start the timer
    start_time = timer()
    # Setup training and save the results
    launch_distillation_trainer(student_model=student_model,
                                teacher_model=teacher_model,
                                loss_type=args.loss_type,
                                pdisco_model=pdisco_model,
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
                                snapshot_path=args.snapshot_dir,
                                grad_norm_clip=args.grad_norm_clip,
                                num_workers=args.num_workers,
                                seed=args.seed,
                                eval_only=args.eval_only,
                                use_ddp=use_ddp,
                                img_size=args.image_size,
                                pdisco_img_size=args.pdisco_image_size,
                                )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    distillation_train_eval()
