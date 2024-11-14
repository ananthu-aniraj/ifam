import fsspec
import os
from dataclasses import asdict
from typing import List, Optional, Tuple, Any, Dict
import copy
from torchvision import transforms as transforms
import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.training_utils.snapshot_class import Snapshot
from utils.wandb_params import init_wandb
from utils.training_utils.ddp_utils import ddp_setup, set_seeds
from utils.training_utils.engine_utils import AverageMeter


class DistilledViTTrainer:
    def __init__(
            self,
            student_model: torch.nn.Module,
            teacher_model: torch.nn.Module,
            loss_type: str,
            pdisco_model: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            batch_size: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            loss_fn: List[torch.nn.Module],
            save_every: int,
            snapshot_path: str,
            loggers: List,
            log_freq: int = 10,
            use_amp: bool = False,
            grad_norm_clip: float = 1.0,
            max_epochs: int = 100,
            num_workers: int = 4,
            eval_only: bool = False,
            use_ddp: bool = False,
            img_size: int = 518,
            pdisco_img_size: int = 518
    ) -> None:
        self._init_ddp(use_ddp)
        self._init_loss_dict()
        loss_fn_train, loss_fn_eval = loss_fn
        self.loss_fn_train = loss_fn_train.to(self.local_rank, non_blocking=True)
        self.loss_fn_eval = loss_fn_eval.to(self.local_rank, non_blocking=True)
        self.loss_type = loss_type
        self.student_model = student_model.to(self.local_rank)
        self.teacher_model = teacher_model.eval().to(self.local_rank)
        if pdisco_model is not None:
            self.pdisco_model = pdisco_model.eval().to(self.local_rank)
        else:
            self.pdisco_model = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.eval_only = eval_only
        self.train_loader = self._prepare_dataloader(train_dataset, num_workers=num_workers)
        self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers)
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.h_fmap = student_model.h_fmap
        self.w_fmap = student_model.w_fmap
        if os.path.isdir(snapshot_path):
            self.is_snapshot_dir = True
        else:
            self.is_snapshot_dir = False
        if loggers:
            if self.local_rank == 0 and self.global_rank == 0:
                loggers[0] = init_wandb(loggers[0])
        self.loggers = loggers
        self.log_freq = log_freq
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.grad_norm_clip = grad_norm_clip
        self.max_epochs = max_epochs
        self.epoch_test_accuracies = []
        self.current_epoch = 0
        self.accum_steps = 1
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        if os.path.isfile(os.path.join(snapshot_path, f"snapshot_best.pt")):
            print("Loading snapshot")
            self._load_snapshot()
        elif os.path.isfile(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()
        self.batch_img_metas = None
        if self.use_ddp:
            print(f"Using DDP with {self.world_size} GPUs")
            self.student_model = DDP(self.student_model, device_ids=[self.local_rank])
        else:
            print("Using single GPU")
        if self.local_rank == 0 and self.global_rank == 0:
            for logger in self.loggers:
                logger.watch(student_model, log="all", log_freq=self.log_freq)
        self.epoch_test_accuracies = []

        if img_size != pdisco_img_size and pdisco_model is not None:
            self.transform_extra = transforms.Resize((img_size, img_size), antialias=True)  # Resize to model input size
        else:
            self.transform_extra = None

    def _init_ddp(self, use_ddp) -> None:
        self.is_slurm_job = "SLURM_NODEID" in os.environ
        self.use_ddp = use_ddp
        if self.is_slurm_job:
            n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            self.global_rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.local_world_size = self.world_size // n_nodes
            self.use_ddp = True
        else:
            if not self.use_ddp:
                self.local_rank = 0
                self.global_rank = 0
                self.world_size = 1
                self.local_world_size = 1
            else:
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.global_rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
                self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    def _init_loss_dict(self) -> None:
        self.loss_dict_train = {'train_loss': AverageMeter()}

        self.loss_dict_val = {'test_loss': AverageMeter()}

    def _prepare_dataloader_ddp(self, dataset: torch.utils.data.Dataset, num_workers: int = 4):

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            sampler=DistributedSampler(dataset)
        )

    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, num_workers: int = 4):
        if self.use_ddp:
            return self._prepare_dataloader_ddp(dataset, num_workers)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )

    def _load_snapshot(self) -> None:
        loc = f"cuda:{self.local_rank}"
        try:
            if self.is_snapshot_dir:
                snapshot = fsspec.open(os.path.join(self.snapshot_path, f"snapshot_best.pt"))
            else:
                snapshot = fsspec.open(self.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location=loc)
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        relevant_keys = [key for key in snapshot.model_state.keys() if "base_model" in key]
        if relevant_keys:
            state_dict = {key.replace('base_model.module.', ''): snapshot.model_state[key] for key in relevant_keys}
        else:
            state_dict = snapshot.model_state
        self.student_model.load_state_dict(state_dict)
        if self.eval_only:
            return
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        self.scheduler.step(snapshot.finished_epoch)
        if snapshot.epoch_test_accuracies is not None:
            self.epoch_test_accuracies = copy.deepcopy(snapshot.epoch_test_accuracies)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, train: bool = True) -> Tuple[Any, Any]:

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                 enabled=self.use_amp):
            maps = self.pdisco_model(source)[1]  # (B, num_parts+1, h_fmap, w_fmap)
            # Check resolution of the feature map vs model resolution
            if maps.shape[-1] != self.w_fmap:
                maps = torch.nn.functional.interpolate(maps, size=(self.h_fmap, self.w_fmap), mode='bilinear',
                                                       align_corners=False)  # (B, num_parts+1, h_fmap, w_fmap)
            argmax_maps = torch.argmax(maps, dim=1)  # (B, h_fmap, w_fmap)
            detected_parts = torch.unique(argmax_maps)  # (num_detected_parts,)
            # Remove background part
            # detected_parts = detected_parts[detected_parts != maps.shape[1] - 1]
            # # Select a random part of the detected parts
            # req_part_idx = torch.randint(0, detected_parts.shape[0], (1,))  # (1,)
            # req_part = detected_parts[req_part_idx]  # (1,)
            # masks = torch.zeros_like(maps, device=argmax_maps.device)
            # # Create binary masks for each part
            # for part in detected_parts:
            #     masks[:, part] = (argmax_maps == part).float()  # (B, num_parts+1, h_fmap, w_fmap)
            #
            # # Construct a new batch with masks
            # masks = masks.flatten(start_dim=2)  # (B, num_parts+1, H*W)
            # masks = masks.flatten(start_dim=0, end_dim=1)  # (B*(num_parts+1), H*W), new batch size is B*(num_parts+1)
            # num_keep = masks.count_nonzero(dim=-1)  # (B*(num_parts+1),)
            # # Create binary mask to indicate part presence
            # part_presence = num_keep > 0
            # mask = masks.float()
            part_idx = detected_parts[torch.randint(0, detected_parts.shape[0], (1,))].item()
            mask = (argmax_maps == part_idx)  # (B, h_fmap, w_fmap)
            mask = mask.flatten(start_dim=1)  # (B, H*W)
            num_keep = mask.count_nonzero(dim=-1)  # (B,)
            # Create binary mask to indicate part presence
            part_presence = num_keep > 0
            mask = mask.float()
            if self.transform_extra:
                source = self.transform_extra(source)
            # Remove elements from batch where part is not present
            mask = mask[part_presence]
            # # Repeat source for each part
            # source = torch.cat(maps.shape[1] * [source], dim=0)  # (B*(num_parts+1), C, H, W)
            source = source[part_presence]
            teacher_outputs = self.teacher_model(source, mask)
            if self.loss_type == 'dino':
                teacher_outputs = self.loss_fn_train.softmax_center_teacher(teacher_outputs)

        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                               enabled=self.use_amp):

            student_outputs = self.student_model(source, mask)  # (B, num_classes)

            if train:
                loss = self.loss_fn_train(student_outputs, teacher_outputs)

                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    # If using AMP, scale the loss, do not call backward on the loss if loss is zero, part presence is zero
                    self.scaler.scale(loss).backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.grad_norm_clip)
                    self.optimizer.step()

            else:
                loss = self.loss_fn_eval(student_outputs, teacher_outputs)

        return student_outputs, loss.item(), part_presence

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if self.use_ddp:
            dataloader.sampler.set_epoch(epoch)
        loss_value = 0
        last_accum_steps = len(dataloader) % self.accum_steps
        updates_per_epoch = (len(dataloader) + self.accum_steps - 1) // self.accum_steps
        num_updates = (epoch - 1) * updates_per_epoch
        last_batch_idx = len(dataloader) - 1
        last_batch_idx_to_accum = len(dataloader) - last_accum_steps

        for key in self.loss_dict_train:
            self.loss_dict_train[key].reset()
        for key in self.loss_dict_val:
            self.loss_dict_val[key].reset()

        for it, batch_data in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = batch_data[0].to(self.local_rank, non_blocking=True)
            batch_preds, batch_loss, part_presence = self._run_batch(source, train=train)
            if train:
                num_updates += 1
                self.scheduler.step_update(num_updates=num_updates)
                self.loss_dict_train['train_loss'].update(batch_loss, batch_preds.size(0))
            else:
                self.loss_dict_val['test_loss'].update(batch_loss, batch_preds.size(0))

            if it % self.log_freq == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} Loss {batch_loss:.5f}")

        if train:
            self.scheduler.step(epoch)
            loss_value = self.loss_dict_train['train_loss'].avg
        else:
            loss_value = self.loss_dict_val['test_loss'].avg
        return loss_value

    def _save_snapshot(self, epoch, save_best: bool = False):
        # capture snapshot
        model = self.student_model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            epoch_test_accuracies=self.epoch_test_accuracies,
        )
        # save snapshot
        snapshot = asdict(snapshot)
        if self.is_snapshot_dir:
            save_path_base = self.snapshot_path
        else:
            save_path_base = os.path.dirname(self.snapshot_path)
        if epoch == self.max_epochs:
            save_path = os.path.join(save_path_base, f"snapshot_final.pt")
        elif save_best:
            save_path = os.path.join(save_path_base, f"snapshot_best.pt")
        else:
            save_path = os.path.join(save_path_base, f"snapshot_{epoch}.pt")

        torch.save(snapshot, save_path)
        print(f"Snapshot saved at epoch {epoch}")

    def finish_logging(self):
        for logger in self.loggers:
            logger.finish()

    def train(self):
        for epoch in range(self.epochs_run, self.max_epochs):
            epoch += 1
            self.current_epoch = epoch
            self.student_model.train()
            train_loss = self._run_epoch(epoch, self.train_loader, train=True)

            logging_dict = {"epoch": epoch,
                            'base_lr': self.optimizer.param_groups[0]['lr'],
                            'scratch_lr': self.optimizer.param_groups[-1]['lr']}

            if self.local_rank == 0 and self.global_rank == 0:
                logging_dict.update({"train_loss": train_loss})
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            elif self.local_rank == 0 and epoch == self.max_epochs:
                self._save_snapshot(epoch)

            # eval run
            if self.test_loader:
                self.student_model.eval()
                test_loss = self._run_epoch(epoch, self.test_loader, train=False)
                if self.local_rank == 0 and self.global_rank == 0:
                    logging_dict.update({"test_loss": test_loss})

            if self.local_rank == 0 and self.global_rank == 0:
                for logger in self.loggers:
                    logger.log(logging_dict)
        if self.local_rank == 0 and self.global_rank == 0:
            self.finish_logging()


def launch_distillation_trainer(student_model: torch.nn.Module,
                                teacher_model: torch.nn.Module,
                                loss_type: str,
                                pdisco_model: torch.nn.Module,
                                train_dataset: torch.utils.data.Dataset,
                                test_dataset: torch.utils.data.Dataset,
                                batch_size: int,
                                optimizer: torch.optim.Optimizer,
                                scheduler: torch.optim.lr_scheduler.LRScheduler,
                                loss_fn: List[torch.nn.Module],
                                epochs: int,
                                save_every: int,
                                loggers: List,
                                log_freq: int,
                                use_amp: bool = False,
                                snapshot_path: str = "snapshot.pt",
                                grad_norm_clip: float = 1.0,
                                num_workers: int = 0,
                                seed: int = 42,
                                eval_only: bool = False,
                                use_ddp: bool = False,
                                img_size: int = 518,
                                pdisco_img_size: int = 518) -> None:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through DistributedTrainer class
     for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    student_model: A PyTorch model to be trained.
    teacher_model: A PyTorch model to be used as a teacher.
    pdisco_model: Trained Pdiscoformer model.
    train_dataset: A DataLoader instance for the model to be trained on.
    test_dataset: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    scheduler: A PyTorch scheduler to adjust the learning rate during training.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    save_every: An integer indicating how often to save the model.
    snapshot_path: A string indicating where to save the model.
    loggers: A list of loggers to log metrics to.
    log_freq: An integer indicating how often to log metrics.
    grad_norm_clip: A float indicating the maximum gradient norm to clip to.
    enable_gradient_clipping: A boolean indicating whether to enable gradient clipping.
    seed: An integer indicating the random seed to use.
    eval_only: A boolean indicating whether to only run evaluation.
    use_ddp: A boolean indicating whether to use DDP.
    @rtype: None
    """

    set_seeds(seed)
    # Loop through training and testing steps for a number of epochs
    if use_ddp:
        ddp_setup()
    trainer = DistilledViTTrainer(student_model=student_model, teacher_model=teacher_model,
                                  loss_type=loss_type,
                                  pdisco_model=pdisco_model,
                                  train_dataset=train_dataset,
                                  test_dataset=test_dataset,
                                  batch_size=batch_size, optimizer=optimizer, scheduler=scheduler,
                                  loss_fn=loss_fn,
                                  save_every=save_every, snapshot_path=snapshot_path, loggers=loggers,
                                  log_freq=log_freq,
                                  use_amp=use_amp,
                                  grad_norm_clip=grad_norm_clip, max_epochs=epochs, num_workers=num_workers,
                                  eval_only=eval_only, use_ddp=use_ddp, img_size=img_size,
                                  pdisco_img_size=pdisco_img_size)

    trainer.train()

    destroy_process_group()
