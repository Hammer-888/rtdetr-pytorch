"""
by lyuwenyu
"""

import time
import json
import datetime

import torch
from PIL import Image

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def fit(
        self,
    ):
        # Fix PIL DecompressionBombError by increasing the limit
        Image.MAX_IMAGE_PIXELS = (
            None  # Remove limit entirely, or set a higher value like 500000000
        )

        print("Start training")
        self.train()

        args = self.cfg

        # Print training configuration
        print("=" * 60)
        print("Training Configuration:")
        print("=" * 60)

        # Print batch size information
        train_batch_size = self.train_dataloader.batch_size
        val_batch_size = self.val_dataloader.batch_size
        print(f"Train Batch Size: {train_batch_size}")
        print(f"Validation Batch Size: {val_batch_size}")

        # Print model structure and parameters
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {n_parameters:,}")
        print(f"Non-trainable parameters: {total_params - n_parameters:,}")

        # Print model architecture overview
        print("\nModel Architecture:")
        print("-" * 40)
        model_to_print = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        print(f"Model Type: {type(model_to_print).__name__}")

        # Print backbone information if available
        if hasattr(model_to_print, "backbone"):
            backbone_type = type(model_to_print.backbone).__name__
            print(f"Backbone: {backbone_type}")

        # Print encoder information if available
        if hasattr(model_to_print, "encoder"):
            encoder_type = type(model_to_print.encoder).__name__
            print(f"Encoder: {encoder_type}")

        # Print decoder information if available
        if hasattr(model_to_print, "decoder"):
            decoder_type = type(model_to_print.decoder).__name__
            print(f"Decoder: {decoder_type}")

        print("=" * 60)
        print()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {
            "epoch": -1,
        }
        best_metric_value = -1  # Track the best metric value

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
            )

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                self.output_dir,
            )

            # Update best statistics
            is_best_epoch = False
            for k in test_stats.keys():
                if k in best_stat:
                    if test_stats[k][0] > best_stat[k]:
                        best_stat["epoch"] = epoch
                        best_stat[k] = test_stats[k][0]
                        is_best_epoch = True
                        best_metric_value = test_stats[k][0]
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]
                    is_best_epoch = True
                    best_metric_value = test_stats[k][0]
            print("best_stat: ", best_stat)

            # Save checkpoints - only keep latest and best
            if self.output_dir:
                # Clean up old checkpoint files first (except checkpoint.pth, checkpoint_latest.pth, checkpoint_best.pth)
                self._cleanup_old_checkpoints()

                # Always save the latest checkpoint
                latest_checkpoint = self.output_dir / "checkpoint_latest.pth"
                dist.save_on_master(self.state_dict(epoch), latest_checkpoint)

                # Save best checkpoint if this is the best epoch so far
                if is_best_epoch:
                    best_checkpoint = self.output_dir / "checkpoint_best.pth"
                    dist.save_on_master(self.state_dict(epoch), best_checkpoint)
                    print(
                        f"New best model saved at epoch {epoch} with metric value {best_metric_value:.4f}"
                    )

                # Also keep the traditional checkpoint.pth as latest for compatibility
                traditional_checkpoint = self.output_dir / "checkpoint.pth"
                dist.save_on_master(self.state_dict(epoch), traditional_checkpoint)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            # Add configuration info to the first epoch log
            if epoch == self.last_epoch + 1:
                model_to_print = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                config_info = {
                    "train_batch_size": self.train_dataloader.batch_size,
                    "val_batch_size": self.val_dataloader.batch_size,
                    "total_parameters": sum(p.numel() for p in self.model.parameters()),
                    "trainable_parameters": n_parameters,
                    "model_type": type(model_to_print).__name__,
                }

                # Add architecture components if available
                if hasattr(model_to_print, "backbone"):
                    config_info["backbone"] = type(model_to_print.backbone).__name__
                if hasattr(model_to_print, "encoder"):
                    config_info["encoder"] = type(model_to_print.encoder).__name__
                if hasattr(model_to_print, "decoder"):
                    config_info["decoder"] = type(model_to_print.decoder).__name__

                log_stats.update(config_info)

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(
        self,
    ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
            self.output_dir,
        )

        if self.output_dir:
            dist.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoint files to save disk space"""
        import glob
        import os

        if not self.output_dir:
            return

        # Files to keep
        keep_files = {"checkpoint.pth", "checkpoint_latest.pth", "checkpoint_best.pth"}

        # Find all checkpoint files
        checkpoint_pattern = str(self.output_dir / "checkpoint*.pth")
        all_checkpoints = glob.glob(checkpoint_pattern)

        for checkpoint_path in all_checkpoints:
            filename = os.path.basename(checkpoint_path)
            if filename not in keep_files:
                try:
                    os.remove(checkpoint_path)
                    print(f"Removed old checkpoint: {filename}")
                except OSError as e:
                    print(f"Warning: Could not remove {filename}: {e}")
