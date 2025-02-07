import fsspec
import os
from dataclasses import asdict
from typing import List, Optional, Tuple, Any, Dict
import copy

import torch
import torchmetrics
from torchmetrics.classification import MultilabelAccuracy, MultilabelAveragePrecision, MultilabelAUROC
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timm.data import Mixup

from timm.utils.model_ema import ModelEmaV3

from utils.training_utils.snapshot_class import Snapshot
from utils.wandb_params import init_wandb
from utils.training_utils.ddp_utils import ddp_setup, set_seeds
from utils.training_utils.engine_utils import AverageMeter

amp_dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported(
    including_emulation=False) else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# note: float16 data type will automatically use a GradScaler
pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[amp_dtype]


class MaskedViTTrainer:
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
            grad_accumulation_steps: int = 1,
            dataset_name: str = "",
            averaging_params: Optional[Dict] = None,
    ) -> None:
        self._init_ddp(use_ddp)
        if dataset_name == "imagenet_a" or dataset_name == "imagenet_r":
            self.num_classes = train_dataset.num_classes
        else:
            self.num_classes = model.num_classes
        # Top-k accuracy metrics for evaluation
        self._init_accuracy_metrics()
        self.dataset_name = dataset_name
        self._init_loss_dict()
        self.h_fmap = model.h_fmap
        self.w_fmap = model.w_fmap
        self.model = model.to(self.local_rank)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.eval_only = eval_only
        self.train_loader = self._prepare_dataloader(train_dataset, num_workers=num_workers, drop_last=True,
                                                     shuffle=True)
        self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers, drop_last=False)
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
        self.grad_norm_clip = grad_norm_clip
        self.max_epochs = max_epochs
        self.mixup_fn = mixup_fn
        self.epoch_test_accuracies = []
        self.current_epoch = 0
        self.accum_steps = grad_accumulation_steps
        # Find Pytorch data type
        if use_amp:
            self.pt_dtype = pt_dtype
            self.amp_dtype = amp_dtype
        else:
            self.amp_dtype = 'float32'
            self.pt_dtype = torch.float32

        # Initialize the GradScaler
        try:
            self.scaler = torch.GradScaler(device="cuda", enabled=(self.amp_dtype == 'float16'))
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == 'float16'))

        self.averaging_params = averaging_params

        if os.path.isfile(os.path.join(snapshot_path, f"snapshot_best.pt")):
            print("Loading snapshot")
            self._load_snapshot()
        elif os.path.isfile(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()
        self.batch_img_metas = None

        self.model_ema = None
        self.averaging_type = averaging_params['type']
        if self.averaging_type == "ema":
            self.model_ema = ModelEmaV3(model, decay=averaging_params['decay'], device=averaging_params['device'],
                                        use_warmup=averaging_params['use_warmup'])

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
        self.per_class_acc_train = {
            'per_class_acc_train': torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes,
                                                                                  average="none").to(
                self.local_rank, non_blocking=True)}
        self.per_class_acc_test = {
            'per_class_acc_test': torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes,
                                                                                 average="none").to(self.local_rank,
                                                                                                    non_blocking=True)}
        if self.num_classes <= 5:
            self.acc_dict_train = {'train_acc': torchmetrics.classification.MulticlassAccuracy(
                num_classes=self.num_classes, top_k=1, average="micro").to(self.local_rank, non_blocking=True),
                                   'macro_avg_acc_top1_train': torchmetrics.classification.MulticlassAccuracy(
                                       num_classes=self.num_classes, top_k=1, average="macro").to(self.local_rank,
                                                                                                  non_blocking=True)}
            self.acc_dict_test = {
                'test_acc': torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, top_k=1,
                                                                           average="micro").to(self.local_rank,
                                                                                               non_blocking=True),
                'macro_avg_acc_top1_test': torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes,
                                                                                          top_k=1, average="macro").to(
                    self.local_rank, non_blocking=True)}
        else:
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
        if self.num_classes == 2:
            self.auroc_train = {
                'auroc_train': torchmetrics.classification.BinaryAUROC(thresholds=None).to(self.local_rank,
                                                                                           non_blocking=True)}
            self.auroc_test = {
                'auroc_test': torchmetrics.classification.BinaryAUROC(thresholds=None).to(self.local_rank,
                                                                                          non_blocking=True)}
        else:
            self.auroc_train = {
                'auroc_train': torchmetrics.classification.MulticlassAUROC(num_classes=self.num_classes,
                                                                           average="macro").to(self.local_rank,
                                                                                               non_blocking=True)}
            self.auroc_test = {
                'auroc_test': torchmetrics.classification.MulticlassAUROC(num_classes=self.num_classes,
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
            sampler=DistributedSampler(dataset),
        )

    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, num_workers: int = 4,
                            drop_last: bool = False, shuffle: bool = False) -> DataLoader:
        if self.use_ddp:
            return self._prepare_dataloader_ddp(dataset, num_workers=num_workers)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    def _load_snapshot(self) -> None:
        loc = f"cuda:{self.local_rank}"
        try:
            if self.is_snapshot_dir:
                snapshot = fsspec.open(os.path.join(self.snapshot_path, f"snapshot_best.pt"))
            else:
                snapshot = fsspec.open(self.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location=loc, weights_only=True)
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
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

    def _run_batch(self, source, targets, inp_mask, train: bool = True, curr_iter: int = 0) -> Tuple[Any, Any]:
        if train and self.use_ddp:
            self.model.require_backward_grad_sync = (
                    (curr_iter + 1) % self.accum_steps == 0 or curr_iter == len(self.train_loader) - 1)

        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=self.pt_dtype):
            # Reshape the input mask to match the output of the model
            downsampled_mask = torch.nn.functional.interpolate(inp_mask.unsqueeze(1).float(),
                                                               size=(self.h_fmap, self.w_fmap),
                                                               mode='nearest-exact').squeeze(
                1).contiguous()  # (B, H, W)
            downsampled_mask = downsampled_mask.flatten(1)  # (B, H*W)

            # Use ema model for evaluation (if available)
            if not train and self.model_ema is not None and self.averaging_params['device'] != "cpu":
                outputs = self.model_ema(source, downsampled_mask)
            else:
                outputs = self.model(source, downsampled_mask)

            if train:
                loss = self.loss_fn_train(outputs, targets)
                loss /= self.accum_steps
            else:
                loss = self.loss_fn_eval(outputs, targets)

        if train:
            self.scaler.scale(loss).backward()
            if (curr_iter + 1) % self.accum_steps == 0 or curr_iter == len(self.train_loader) - 1:
                if self.grad_norm_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        return outputs, loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if self.use_ddp:
            dataloader.sampler.set_epoch(epoch)

        last_accum_steps = len(dataloader) % self.accum_steps
        updates_per_epoch = len(dataloader) // self.accum_steps + (1 if last_accum_steps > 0 else 0)
        num_updates = (epoch - 1) * updates_per_epoch
        last_batch_idx = len(dataloader) - 1

        for key in self.loss_dict_train:
            self.loss_dict_train[key].reset()
        for key in self.loss_dict_val:
            self.loss_dict_val[key].reset()
        accuracies_dict = {}
        accuracies_dict_per_class = {}
        for key in self.per_class_acc_train.keys():
            self.per_class_acc_train[key].reset()
        for key in self.per_class_acc_test.keys():
            self.per_class_acc_test[key].reset()
        for key in self.auroc_train.keys():
            self.auroc_train[key].reset()
        for key in self.auroc_test.keys():
            self.auroc_test[key].reset()

        # Compute metrics for evaluation
        for key in self.acc_dict_train.keys():
            self.acc_dict_train[key].reset()
        for key in self.acc_dict_test.keys():
            self.acc_dict_test[key].reset()

        for it, mini_batch in enumerate(dataloader):
            source = mini_batch[0]
            targets = mini_batch[1]
            inp_mask = mini_batch[2]
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank, non_blocking=True)
            targets = targets.to(self.local_rank, non_blocking=True)
            inp_mask = inp_mask.to(self.local_rank, non_blocking=True)

            if train and self.mixup_fn is not None:
                source, targets = self.mixup_fn(source, targets)
            batch_preds, batch_loss = self._run_batch(source, targets, inp_mask, train, curr_iter=it)
            if self.dataset_name == "imagenet_a" or self.dataset_name == "imagenet_r":
                batch_preds = batch_preds[:, self.test_dataset.class_mask]
                batch_preds_stage_1 = batch_preds_stage_1[:, self.test_dataset.class_mask]

            if train:
                if (it + 1) % self.accum_steps == 0 or it == last_batch_idx:
                    num_updates += 1
                    self.scheduler.step_update(num_updates=num_updates)
                    if self.model_ema is not None:
                        self.model_ema.update(self.model, step=num_updates)
                self.loss_dict_train['train_loss'].update(batch_loss, source.size(0))
                if self.mixup_fn is None:
                    for key in self.acc_dict_train.keys():
                        self.acc_dict_train[key].update(batch_preds, targets.long())

                    for key in self.per_class_acc_train.keys():
                        self.per_class_acc_train[key].update(batch_preds, targets)
                    if self.num_classes == 2:
                        with torch.no_grad():
                            probs_positives = torch.softmax(batch_preds, dim=1)[:, 1].squeeze()  # (B,)
                        self.auroc_train['auroc_train'].update(probs_positives, targets.long())
                    else:
                        self.auroc_train['auroc_train'].update(batch_preds, targets.long())
            else:
                self.loss_dict_val['test_loss'].update(batch_loss, source.size(0))
                for key in self.acc_dict_test.keys():
                    self.acc_dict_test[key].update(batch_preds, targets.long())

                for key in self.per_class_acc_test.keys():
                    self.per_class_acc_test[key].update(batch_preds, targets)
                if self.num_classes == 2:
                    probs_positives = torch.softmax(batch_preds, dim=1)[:, 1].squeeze()
                    self.auroc_test['auroc_test'].update(probs_positives, targets.long())
                else:
                    self.auroc_test['auroc_test'].update(batch_preds, targets.long())

            if it % self.log_freq == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} Loss {batch_loss:.5f}")

        if train:
            loss_value = self.loss_dict_train['train_loss'].avg
            if self.mixup_fn is None:
                for key in self.acc_dict_train.keys():
                    accuracies_dict[key] = self.acc_dict_train[key].compute().item() * 100

                for key in self.per_class_acc_train.keys():
                    accuracies_dict_per_class[key] = self.per_class_acc_train[key].compute() * 100
                    for i in range(self.num_classes):
                        accuracies_dict_per_class[key][i] = accuracies_dict_per_class[key][i].item()
                for key in self.auroc_train.keys():
                    accuracies_dict[key] = self.auroc_train[key].compute().item() * 100
            self.scheduler.step(epoch)
        else:
            loss_value = self.loss_dict_val['test_loss'].avg
            for key in self.acc_dict_test.keys():
                accuracies_dict[key] = self.acc_dict_test[key].compute().item() * 100

            for key in self.per_class_acc_test.keys():
                accuracies_dict_per_class[key] = self.per_class_acc_test[key].compute() * 100
                for i in range(self.num_classes):
                    accuracies_dict_per_class[key][i] = accuracies_dict_per_class[key][i].item()
            for key in self.auroc_test.keys():
                accuracies_dict[key] = self.auroc_test[key].compute().item() * 100
        return loss_value, accuracies_dict, accuracies_dict_per_class

    def _save_snapshot(self, epoch, save_best: bool = False):
        # capture snapshot
        if self.model_ema is not None:
            model = self.model_ema.module
        else:
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
            train_loss, acc_dict_train, _ = self._run_epoch(epoch, self.train_loader, train=True)

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
                test_loss, acc_dict_test, _ = self._run_epoch(epoch, self.test_loader, train=False)
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
                test_loss, acc_dict_test, per_class_acc_dict = self._run_epoch(0, self.test_loader, train=False)

            print(f'Test loss: {test_loss:.5f}'
                  f'| Test acc: {acc_dict_test["test_acc"]:.5f} '
                  f'| Macro avg acc top1: {acc_dict_test["macro_avg_acc_top1_test"]:.5f}'
                  f'| AUROC: {acc_dict_test["auroc_test"]:.5f}')
            # Print per class accuracy if num_classes <= 5
            if self.num_classes <= 5:
                for key in per_class_acc_dict.keys():
                    for i in range(self.num_classes):
                        print(f'Class {i} | {key}: {per_class_acc_dict[key][i]:.5f}')

        if self.local_rank == 0 and self.global_rank == 0:
            logging_dict.update({"test_loss": test_loss})
            logging_dict.update(acc_dict_test)
            for logger in self.loggers:
                logger.log(logging_dict)
        self.finish_logging()


def launch_masked_vit_trainer(model: torch.nn.Module,
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
                              grad_accumulation_steps: int = 1,
                              dataset_name: str = "",
                              averaging_params: Optional[Dict] = None,
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
    ema_params: A dictionary of parameters for EMA.
    """

    set_seeds(seed)
    # Loop through training and testing steps for a number of epochs
    if use_ddp:
        ddp_setup()
    trainer = MaskedViTTrainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset,
                               batch_size=batch_size, optimizer=optimizer, scheduler=scheduler,
                               loss_fn=loss_fn,
                               save_every=save_every, snapshot_path=snapshot_path, loggers=loggers,
                               log_freq=log_freq,
                               use_amp=use_amp,
                               grad_norm_clip=grad_norm_clip, max_epochs=epochs, num_workers=num_workers,
                               mixup_fn=mixup_fn, eval_only=eval_only, use_ddp=use_ddp,
                               grad_accumulation_steps=grad_accumulation_steps,
                               dataset_name=dataset_name,
                               averaging_params=averaging_params)
    if eval_only:
        trainer.test_only()
    else:
        trainer.train()
    if use_ddp:
        destroy_process_group()
