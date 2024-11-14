import fsspec
import os
from dataclasses import asdict
from typing import List, Optional, Tuple, Any
import copy

import torch
import torchmetrics
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timm.data import Mixup

from utils.training_utils.snapshot_class import Snapshot
from utils.wandb_params import init_wandb
from utils.data_utils.class_balanced_distributed_sampler import ClassBalancedDistributedSampler
from utils.data_utils.class_balanced_sampler import ClassBalancedRandomSampler
from utils.training_utils.ddp_utils import ddp_setup, set_seeds
from utils.training_utils.engine_utils import AverageMeter


class BaselineTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
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
            mixup_fn: Optional[Mixup] = None,
            eval_only: bool = False,
            use_ddp: bool = False,
    ) -> None:
        self._init_ddp(use_ddp)
        self.num_classes = model.num_classes
        # Top-k accuracy metrics for evaluation
        self._init_accuracy_metrics()
        self._init_loss_dict()
        self.model = model.to(self.local_rank)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.eval_only = eval_only
        self.train_loader = self._prepare_dataloader(train_dataset, num_workers=num_workers)
        self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers)
        if len(loss_fn) == 1:
            self.loss_fn_train = self.loss_fn_eval = loss_fn[0]
        else:
            self.loss_fn_train = loss_fn[0]
            self.loss_fn_eval = loss_fn[1]
        self.loss_fn_eval = self.loss_fn_eval.to(self.local_rank, non_blocking=True)
        self.loss_fn_train = self.loss_fn_train.to(self.local_rank, non_blocking=True)
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
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
        self.mixup_fn = mixup_fn
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
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            print("Using single GPU")
        self.epoch_test_accuracies = []
        if self.local_rank == 0 and self.global_rank == 0:
            for logger in self.loggers:
                logger.watch(model, log="all", log_freq=self.log_freq)

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

    def _init_accuracy_metrics(self) -> None:
        self.acc_dict_train = {'train_acc': torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes, top_k=1,
            average="micro").to(self.local_rank,
                                non_blocking=True),
                               'train_acc_top5': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="micro").to(self.local_rank,
                                                       non_blocking=True),
                               'macro_avg_acc_top1_train': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=1,
                                   average="macro").to(self.local_rank,
                                                       non_blocking=True),
                               'macro_avg_acc_top5_train': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="macro").to(self.local_rank,
                                                       non_blocking=True)}

        self.acc_dict_test = {'test_acc': torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes, top_k=1,
            average="micro").to(self.local_rank,
                                non_blocking=True),
                              'test_acc_top5': torchmetrics.classification.MulticlassAccuracy(
                                  num_classes=self.num_classes, top_k=5,
                                  average="micro").to(self.local_rank,
                                                      non_blocking=True),
                              'macro_avg_acc_top1_test': torchmetrics.classification.MulticlassAccuracy(
                                  num_classes=self.num_classes, top_k=1,
                                  average="macro").to(self.local_rank,
                                                      non_blocking=True),
                              'macro_avg_acc_top5_test': torchmetrics.classification.MulticlassAccuracy(
                                  num_classes=self.num_classes, top_k=5,
                                  average="macro").to(self.local_rank,
                                                      non_blocking=True)}

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
        self.model.load_state_dict(state_dict)
        if self.eval_only:
            return
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        self.scheduler.step(snapshot.finished_epoch)
        if snapshot.epoch_test_accuracies is not None:
            self.epoch_test_accuracies = copy.deepcopy(snapshot.epoch_test_accuracies)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, train: bool = True) -> Tuple[Any, Any]:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                               enabled=self.use_amp):

            outputs = self.model(source)

            if train:
                loss = self.loss_fn_train(outputs, targets)
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    self.optimizer.step()
            else:
                loss = self.loss_fn_eval(outputs, targets)

        return outputs, loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if self.use_ddp:
            dataloader.sampler.set_epoch(epoch)

        last_accum_steps = len(dataloader) % self.accum_steps
        updates_per_epoch = (len(dataloader) + self.accum_steps - 1) // self.accum_steps
        num_updates = (epoch - 1) * updates_per_epoch
        last_batch_idx = len(dataloader) - 1
        last_batch_idx_to_accum = len(dataloader) - last_accum_steps

        for key in self.loss_dict_train:
            self.loss_dict_train[key].reset()
        for key in self.loss_dict_val:
            self.loss_dict_val[key].reset()
        accuracies_dict = {}
        # Compute metrics for evaluation
        for key in self.acc_dict_train.keys():
            self.acc_dict_train[key].reset()
        for key in self.acc_dict_test.keys():
            self.acc_dict_test[key].reset()

        for it, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank, non_blocking=True)
            targets = targets.to(self.local_rank, non_blocking=True)

            if train and self.mixup_fn is not None:
                source, targets = self.mixup_fn(source, targets)
            batch_preds, batch_loss = self._run_batch(source, targets, train)
            if train:
                num_updates += 1
                self.scheduler.step_update(num_updates=num_updates)
                self.loss_dict_train['train_loss'].update(batch_loss, source.size(0))
                if self.mixup_fn is None:
                    for key in self.acc_dict_train.keys():
                        self.acc_dict_train[key].update(batch_preds, targets)
            else:
                self.loss_dict_val['test_loss'].update(batch_loss, source.size(0))
                for key in self.acc_dict_test.keys():
                    self.acc_dict_test[key].update(batch_preds, targets)

            if it % self.log_freq == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} Loss {batch_loss:.5f}")

        if train:
            self.scheduler.step(epoch)
            loss_value = self.loss_dict_train['train_loss'].avg
            if self.mixup_fn is None:
                for key in self.acc_dict_train.keys():
                    accuracies_dict[key] = self.acc_dict_train[key].compute().item() * 100
        else:
            loss_value = self.loss_dict_val['test_loss'].avg
            for key in self.acc_dict_test.keys():
                accuracies_dict[key] = self.acc_dict_test[key].compute().item() * 100
        return loss_value, accuracies_dict

    def _save_snapshot(self, epoch, save_best: bool = False):
        # capture snapshot
        model = self.model
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
            self.model.train()
            train_loss, acc_dict_train = self._run_epoch(epoch, self.train_loader, train=True)

            logging_dict = {"epoch": epoch,
                            'base_lr': self.optimizer.param_groups[0]['lr'],
                            'scratch_lr': self.optimizer.param_groups[-1]['lr']}

            if self.local_rank == 0 and self.global_rank == 0:
                logging_dict.update({"train_loss": train_loss})
                logging_dict.update(acc_dict_train)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            elif self.local_rank == 0 and epoch == self.max_epochs:
                self._save_snapshot(epoch)

            # eval run
            if self.test_loader:
                self.model.eval()
                test_loss, acc_dict_test = self._run_epoch(epoch, self.test_loader, train=False)
                if self.local_rank == 0 and self.global_rank == 0:
                    test_acc = acc_dict_test['test_acc']
                    self.epoch_test_accuracies.append(test_acc)
                    max_acc = max(self.epoch_test_accuracies)
                    max_acc_index = self.epoch_test_accuracies.index(max_acc)
                    if max_acc_index == len(self.epoch_test_accuracies) - 1:
                        self._save_snapshot(epoch, save_best=True)

                    logging_dict.update({"test_loss": test_loss
                                         })
                    logging_dict.update(acc_dict_test)
                    for logger in self.loggers:
                        logger.log(logging_dict)
        if self.local_rank == 0 and self.global_rank == 0:
            self.finish_logging()

    def test_only(self):
        self.model.eval()
        logging_dict = {"epoch": 0}
        with torch.inference_mode():
            if self.test_loader:
                test_loss, acc_dict_test = self._run_epoch(0, self.test_loader, train=False)
            print(f'Test loss: {test_loss:.5f}'
                  f'| Test acc: {acc_dict_test["test_acc"]:.5f} '
                  f'| Test acc top5: {acc_dict_test["test_acc_top5"]:.5f} '
                  f'| Macro avg acc top1: {acc_dict_test["macro_avg_acc_top1_test"]:.5f} '
                  f'| Macro avg acc top5: {acc_dict_test["macro_avg_acc_top5_test"]:.5f}')
        if self.local_rank == 0 and self.global_rank == 0:
            logging_dict.update({"test_loss": test_loss})
            logging_dict.update(acc_dict_test)
            for logger in self.loggers:
                logger.log(logging_dict)
        self.finish_logging()


def launch_baseline_trainer(model: torch.nn.Module,
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
                            mixup_fn: Optional[Mixup] = None,
                            seed: int = 42,
                            eval_only: bool = False,
                            use_ddp: bool = False,
                            ) -> None:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through DistributedTrainer class
     for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
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
    mixup_fn: A Mixup instance to apply mixup to the training data.
    seed: An integer indicating the random seed to use.
    eval_only: A boolean indicating whether to only run evaluation.
    use_ddp: A boolean indicating whether to use DDP.
        class_balanced_sampling: A boolean indicating whether to use class-balanced sampling
    num_samples_per_class: An integer indicating the number of samples per class for class-balanced sampling
    @rtype: None
    """

    set_seeds(seed)
    # Loop through training and testing steps for a number of epochs
    if use_ddp:
        ddp_setup()
    trainer = BaselineTrainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset,
                              batch_size=batch_size, optimizer=optimizer, scheduler=scheduler,
                              loss_fn=loss_fn,
                              save_every=save_every, snapshot_path=snapshot_path, loggers=loggers,
                              log_freq=log_freq,
                              use_amp=use_amp,
                              grad_norm_clip=grad_norm_clip, max_epochs=epochs, num_workers=num_workers,
                              mixup_fn=mixup_fn, eval_only=eval_only, use_ddp=use_ddp)
    if eval_only:
        trainer.test_only()
    else:
        trainer.train()

    destroy_process_group()
