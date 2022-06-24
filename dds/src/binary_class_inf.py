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
            pos_list.append(preds[:len(preds)//2])
            pos_target.append(targets[:len(preds)//2])
            neg_list.append(preds[len(preds)//2:])
            neg_target.append(targets[len(preds)//2:])
            targets_list.append(targets)

    cls_types = np.concatenate(cls_list).squeeze(1)
    predic = np.concatenate(preds_list)
    y_test = np.concatenate(targets_list)
    pred = (predic >= 0.5).astype(np.int64)
    
    edges = np.arange(10 + 1) / 10
    g_pos = np.abs(np.concatenate(pos_list) - 1.0)
    g_neg = np.abs(np.concatenate(neg_list))
    pos_number = []
    neg_number = []
    for i in range(10):
        pos_inds = (g_pos >= edges[i]) & (g_pos < edges[i + 1])
        neg_inds = (g_neg >= edges[i]) & (g_neg < edges[i + 1])
        pos_num_in_bin = sum(pos_inds)
        neg_num_in_bin = sum(neg_inds)
        pos_number.append(pos_num_in_bin)
        neg_number.append(neg_num_in_bin)
    print(pos_number, sum(pos_number), [round(i, 2) for i in pos_number/sum(pos_number)])
    print(neg_number, sum(neg_number), [round(i, 2) for i in neg_number/sum(neg_number)])
    
    # with open('tg_c.txt', 'w') as fw:
    #     for p, y, f, c in zip(pred, y_test, predic, cls_types):
    #         fw.writelines(str(p) + ' ' + str(y) + ' ' + str(f) + ' ' + str(c) + '\n')

    posd = (np.concatenate(pos_list) >= 0.5).astype(np.int64)
    negd = (np.concatenate(neg_list) >= 0.5).astype(np.int64)
    post = np.concatenate(pos_target)
    negt = np.concatenate(neg_target)
    pos_acc = sk_metrics.accuracy_score(post, posd)
    neg_acc = sk_metrics.accuracy_score(negt, negd)
    print(f"pos_acc: {pos_acc}, neg_acc: {neg_acc}")

    pos_error_cls_types = cls_types[posd != post]
    neg_error_cls_types = cls_types[negd != negt]

    from collections import Counter
    pos_counter = Counter(pos_error_cls_types.tolist())
    neg_counter = Counter(neg_error_cls_types.tolist())

    print(f"pos_counter{pos_counter}")
    print(f"neg_counter{neg_counter}")

    dict_cls_types = dict(sorted(Counter(cls_types).items()))
    dict_pos_err_cls_types = dict(sorted(Counter(pos_error_cls_types.tolist()).items()))
    dict_neg_err_cls_types = dict(sorted(Counter(neg_error_cls_types.tolist()).items()))
    
    pos_err_abs = []
    neg_err_abs = []
    target_abs = []

    pos_err_rate = []
    neg_err_rate = []
    all_rate = []

    for i in range(86):
        if i in dict_cls_types.keys():
            target_abs.append(dict_cls_types[i])
        else:
            target_abs.append(0)
        
        if i in dict_pos_err_cls_types.keys():
            pos_err_abs.append(dict_pos_err_cls_types[i])
        else:
            pos_err_abs.append(0)
        
        if i in dict_neg_err_cls_types.keys():
            neg_err_abs.append(dict_neg_err_cls_types[i])
        else:
            neg_err_abs.append(0)
        
        pos_err_rate.append(pos_err_abs[-1] / (target_abs[-1] + 1e-10))
        neg_err_rate.append(neg_err_abs[-1] / (target_abs[-1] + 1e-10))
        all_rate.append((pos_err_rate[-1] + neg_err_rate[-1]) / 2)

    
    # plot
    save_dir = os.path.dirname(cfg.checkpoint.restore_file)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(np.arange(86), pos_err_abs)
    plt.xlabel('Relations')
    plt.ylabel('Error predictions')
    plt.title('pos_err_abs_test')
    plt.savefig(os.path.join(save_dir, 'pos_err_abs_test.png'))

    plt.figure()
    plt.bar(np.arange(86), neg_err_abs)
    plt.xlabel('Relations')
    plt.ylabel('Error predictions')
    plt.title('neg_err_abs_test')
    plt.savefig(os.path.join(save_dir, 'neg_err_abs_test.png'))

    plt.figure()
    plt.bar(np.arange(86), target_abs)
    plt.xlabel('Relations')
    plt.ylabel('Numbers')
    plt.title('target_abs_test')
    plt.savefig(os.path.join(save_dir, 'target_abs_test.png'))
    
    plt.figure()
    plt.bar(np.arange(86), pos_err_rate)
    plt.xlabel('Relations')
    plt.ylabel('Error predictions rate')
    plt.title('pos_err_rate_test')
    plt.savefig(os.path.join(save_dir, 'pos_err_rate_test.png'))

    plt.figure()
    plt.bar(np.arange(86), neg_err_rate)
    plt.xlabel('Relations')
    plt.ylabel('Error predictions rate')
    plt.title('neg_err_rate_test')
    plt.savefig(os.path.join(save_dir, 'neg_err_rate_test.png'))
    
    plt.figure()
    plt.bar(np.arange(86), all_rate)
    plt.xlabel('Relations')
    plt.ylabel('Error predictions rate')
    plt.title('all_rate_test')
    plt.savefig(os.path.join(save_dir, 'all_rate_test.png'))

    # save csv
    import pandas as pd
    df = pd.DataFrame({'target_abs': target_abs, 'pos_err_abs': pos_err_abs, 'neg_err_abs': neg_err_abs, 'pos_err_rate': pos_err_rate, 'neg_err_rate': neg_err_rate, 'all_rate': all_rate})
    df.to_csv(os.path.join(save_dir, 'results_test.csv'))

    acc = sk_metrics.accuracy_score(y_test, pred)
    auc_roc = sk_metrics.roc_auc_score(y_test, predic)
    f1_score = sk_metrics.f1_score(y_test, pred)
    p, r, t = sk_metrics.precision_recall_curve(y_test, predic)
    auc_prc = sk_metrics.auc(r, p)
    p = sk_metrics.average_precision_score(y_test, predic)
    print(f"acc: {acc}, auc_roc: {auc_roc}, auc_prc: {auc_prc}, f1_score: {f1_score}, p: {p}")

    
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
