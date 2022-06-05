import imp
import logging
import os
import numpy as np
import torch
from fairseq.data import (
    IdDataset,
    NestedDictionaryDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    NumSamplesDataset,
    NumelDataset,
    data_utils,
    LeftPadDataset,
    BaseWrapperDataset,
)
from fairseq.data.shorten_dataset import TruncateDataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II
from fairseq.data.indexed_dataset import (
    MMapIndexedDataset,
    get_available_dataset_impl,
    make_dataset,
    infer_dataset_impl,
)
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from fairseq.data.molecule.molecule import Tensor2Data
from fairseq.tasks.doublemodel import NoiseOrderedDataset, StripTokenDatasetSizes
from fairseq.data.append_token_dataset import AppendTokenDataset


logger = logging.getLogger(__name__)

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (1048567, rlimit[1]))


@dataclass
class BinaryClassConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    num_classes: int = field(default=2)
    scaler_label: bool = field(default=False)
    no_shuffle: bool = field(default=False)
    shorten_method: ChoiceEnum(["none", "truncate", "random_crop"]) = field(default="truncate")
    shorten_data_split_list: str = field(default="")
    max_positions: int = II("model.max_positions")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II("dataset.dataset_impl")
    seed: int = II("common.seed")
    order_noise: int = field(default=5)


@register_task("binary_class_task", dataclass=BinaryClassConfig)
class BinaryClassTask(FairseqTask):

    cfg: BinaryClassConfig

    def __init__(self, cfg: BinaryClassConfig, data_dictionary, label_dictionary, cell_dictionary):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self.dictionary.add_symbol("[MASK]")
        self.label_dictionary = label_dictionary
        self.cell_dictionary = cell_dictionary
        self._max_positions = cfg.max_positions
        self.seed = cfg.seed
        self.order_noise = cfg.order_noise

        self.drug_a = cfg.source_lang
        self.drug_b = cfg.target_lang

    @classmethod
    def setup_task(cls, cfg: BinaryClassConfig, **kwargs):
        
        assert cfg.num_classes > 0
        data_dict = cls.load_dictionary(os.path.join(cfg.data, "dict.{}.txt".format(cfg.source_lang)))
        logger.info(
            "[input] Dictionary {}: {} types.".format(
                os.path.join(cfg.data), len(data_dict)
            )
        )
        label_dict = cls.load_dictionary(os.path.join(cfg.data, "label", "dict.txt"))
        logger.info(
            "[label] Dictionary {}: {} types.".format(
                os.path.join(cfg.data, "label"), len(label_dict)
            )
        )
        cell_dict = cls.load_dictionary(os.path.join(cfg.data, 'cell', 'dict.txt'))
        logger.info(
            "[cell] Dictionary {}: {} types.".format(
                os.path.join(cfg.data, "label"), len(cell_dict)
            )
        )

        return cls(cfg, data_dict, label_dict, cell_dict)

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        dataset = self.load_dataset_src_tgt(split)
        logger.info("Loaded {} with #samples: {}.".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def get_path(self, key, split):
        return os.path.join(self.cfg.data, key, split)
    
    def load_dataset_src_tgt(self, split):
        # load drug a and b (postive sample)
        drug_a_prefix = self.get_path("", '{}.{}-{}.{}'.format(split, self.drug_a, self.drug_b, self.drug_a))
        drug_b_prefix = self.get_path("", '{}.{}-{}.{}'.format(split, self.drug_a, self.drug_b, self.drug_b))
        if not MMapIndexedDataset.exists(drug_a_prefix):
            raise FileNotFoundError("SMILES data {} not found.".format(drug_a_prefix))
        if not MolMMapIndexedDataset.exists(drug_a_prefix):
            raise FileNotFoundError("PyG data {} not found.".format(drug_a_prefix))
        if not MMapIndexedDataset.exists(drug_b_prefix):
            raise FileNotFoundError("SMILES data {} not found.".format(drug_b_prefix))
        if not MolMMapIndexedDataset.exists(drug_b_prefix):
            raise FileNotFoundError("PyG data {} not found.".format(drug_b_prefix))
        
        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(drug_a_prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        drug_a_dataset = make_dataset(drug_a_prefix, impl=dataset_impl)
        assert drug_a_dataset is not None

        drug_b_dataset = make_dataset(drug_b_prefix, impl=dataset_impl)
        assert drug_b_dataset is not None
        
        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(drug_a_dataset))

        drug_a_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDatasetSizes(drug_a_dataset, self.source_dictionary.eos()),
                self._max_positions - 1,
            ),
            self.source_dictionary.eos(),
        )
        drug_b_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDatasetSizes(drug_b_dataset, self.target_dictionary.eos()),
                self._max_positions - 1,
            ),
            self.target_dictionary.eos(),
        )

        src_dataset_graph = make_graph_dataset(drug_a_prefix, impl=dataset_impl)
        assert src_dataset_graph is not None
        src_dataset_graph = Tensor2Data(src_dataset_graph)

        tgt_dataset_graph = make_graph_dataset(drug_b_prefix, impl=dataset_impl)
        assert tgt_dataset_graph is not None
        tgt_dataset_graph = Tensor2Data(tgt_dataset_graph)

        # load drug neg a and neg b (negtivate sample) ##END##
        dataset = {
            "id": IdDataset(),
            "drug_a_seq": {
                "src_tokens": LeftPadDataset(drug_a_dataset, pad_idx=self.source_dictionary.pad()),
                "src_lengths": NumelDataset(drug_a_dataset),
            },
            "drug_b_seq": {
                "src_tokens": LeftPadDataset(drug_b_dataset, pad_idx=self.target_dictionary.pad()),
                "src_lengths": NumelDataset(drug_b_dataset),
            },
            "drug_a_graph": {"graph": src_dataset_graph,},
            "drug_b_graph": {"graph": tgt_dataset_graph,},
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(drug_a_dataset, reduce=True),
        }

        prefix_label = self.get_path("label", split)
        label_dataset = make_dataset(prefix_label, impl=dataset_impl)
        assert label_dataset is not None

        dataset.update(
            label=OffsetTokensDataset(
                StripTokenDataset(label_dataset, id_to_strip=self.label_dictionary.eos()),
                offset=-self.label_dictionary.nspecial,
            )
        )

        prefix_cell = self.get_path('cell', split)
        cell_dataset = make_dataset(prefix_cell, impl=dataset_impl)
        assert cell_dataset is not None

        dataset.update(
            cell=OffsetTokensDataset(
                StripTokenDataset(cell_dataset, id_to_strip=self.cell_dictionary.eos()),
                offset=-self.cell_dictionary.nspecial,
            )
        )
        
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[drug_a_dataset.sizes])
        dataset = NoiseOrderedDataset(
            nested_dataset,
            sort_order=[shuffle, drug_a_dataset.sizes],
            seed=self.seed,
            order_noise=self.order_noise,
        )
        return dataset

    def build_model(self, cfg):
        model = super().build_model(cfg)
        model.register_classification_head(
            getattr(cfg, "classification_head_name", "sentence_classification_head"),
            num_classes=self.cfg.num_classes,
        )
        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def ddi_inference_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            preds, targets, clstypes = criterion.forward_inference(model, sample)
        return preds, targets, clstypes

class TruncateSizesDataset(BaseWrapperDataset):
    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        self.truncation_length = truncation_length

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)