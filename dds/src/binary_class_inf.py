#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch

sys.path.append(os.getcwd())

from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics as sk_metrics
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    # logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)
        
    trainer = Trainer(cfg, task, model, criterion)
    checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    
    validate(cfg, trainer, task)


# TODO
def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(0)
    subset = cfg.dataset.valid_subset
    logger.info('begin validation on "{}" subset'.format(subset))

    # Initialize data iterator
    itr = trainer.get_valid_iterator(subset).next_epoch_itr(
        shuffle=False, set_dataset_epoch=False  # use a fixed valid set
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=0,
        prefix=f"valid on '{subset}' subset",
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
    )

    model = trainer.model
    criterion = trainer.criterion
    preds_list = []
    targets_list = []

    pos_list = []
    neg_list = []
    pos_target = []
    neg_target = []
    cls_list = []
    
    for i, sample in enumerate(tqdm(progress)):
        if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
            break
        with torch.no_grad():
            model.eval()
            criterion.eval()
            sample, _ = trainer._prepare_sample(sample)
            preds, targets, cls_type = task.ddi_inference_step(sample, trainer.model, trainer.criterion)
            preds_list.append(preds)
            cls_list.append(cls_type)
            targets_list.append(targets)

    predic = np.concatenate(preds_list)
    y_test = np.concatenate(targets_list)
    pred = (predic >= 0.5).astype(np.int64)
    
    acc = sk_metrics.accuracy_score(y_test, pred)
    auc_roc = sk_metrics.roc_auc_score(y_test, predic)
    f1_score = sk_metrics.f1_score(y_test, pred)

    p, r, t = sk_metrics.precision_recall_curve(y_test, predic)
    auc_prc = sk_metrics.auc(r, p)

    bacc = sk_metrics.balanced_accuracy_score(y_test, pred)
    kappa = sk_metrics.cohen_kappa_score(y_test, pred)
    prec = sk_metrics.precision_score(y_test, pred)
    recall = sk_metrics.recall_score(y_test, pred)

    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_test, pred).ravel()
    TPR = tp / (tp + fn)

    print(f"acc: {acc}, bacc: {bacc}, auc_roc: {auc_roc}, auc_prc: {auc_prc}, prec: {prec}, \
        recall: {recall}, f1_score: {f1_score}, TPR: {TPR}, kappa: {kappa}")

    
def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
