from dataclasses import dataclass, field
import imp
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
import math
import torch.nn.functional as F
from fairseq import metrics
from omegaconf import II
import numpy as np
import pdb

@dataclass
class BinaryClassConfig(FairseqDataclass):
    classification_head_name: str = II("model.classification_head_name")
    consis_alpha: float = field(default=0.0)
    mt_alpha: float = field(default=1.0)
    p_consis_alpha: float = field(default=0.0)

@register_criterion("binary_class_loss_rank", dataclass=BinaryClassConfig)
class BinaryClassRankCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, consis_alpha, mt_alpha, p_consis_alpha):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.consis_alpha = consis_alpha
        self.mt_alpha = mt_alpha
        self.p_consis_alpha = p_consis_alpha
        acc_sum = torch.zeros(30)
        self.register_buffer('acc_sum', acc_sum)

    def build_input(self, sample, classification_head_name):
        return {
            'drug_a_seq': sample['drug_a_seq'] if 'drug_a_seq' in  sample else None,
            'drug_b_seq': sample['drug_b_seq'] if 'drug_b_seq' in  sample else None,
            'drug_a_graph': sample['drug_a_graph'] \
                if "drug_a_graph" in sample else None,
            'drug_b_graph': sample['drug_b_graph'] \
                if "drug_b_graph" in sample else None,
            'cell_line': sample['cell'],
            'features_only': True,
            'classification_head_name': classification_head_name,
            }

    def forward(self, model, sample, reduce=True):

        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        input = self.build_input(sample, self.classification_head_name)
        logits = model(**input)

        labels = model.get_targets(sample['label'], None).view(-1)
        sample_size = labels.size(0)

        # pdb.set_trace()
        pos_logits = logits[labels == 1]
        neg_logits = logits[labels == 0]

        # pos_weights = (neg_logits.size(0)) / (pos_logits.size(0) + neg_logits.size(0) + 1e-8)
        # neg_weights = (pos_logits.size(0)) / (pos_logits.size(0) + neg_logits.size(0) + 1e-8)
        # pos_weights, neg_weights = 1, 1
        margin = 1
        if len(pos_logits) == 0:
            loss = -1 * neg_logits.mean() + margin
        else:
            loss = -1 * (pos_logits.mean() - neg_logits.mean()) + margin
        # print(loss)
        pos_preds = torch.sigmoid(pos_logits).detach()
        neg_preds = torch.sigmoid(neg_logits).detach()

        logging_out = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
            "n_pos": pos_logits.size(0),
            "n_neg": neg_logits.size(0),
            "logits": logits.squeeze().data,
            "labels": labels.data

        }
        logging_out["ncorrect"] = (pos_preds >= 0.5).sum() + (neg_preds < 0.5).sum()
        logging_out["pos_acc"] = (pos_preds >= 0.5).sum() 
        logging_out["neg_acc"] = (neg_preds < 0.5).sum()

        return loss, sample_size, logging_out

    def forward_inference(self, model, sample, reduce=True):
        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        input = self.build_input(sample, self.classification_head_name)
        logits = model(**input)
        
        preds = []

        labels = model.get_targets(sample['label'], None).view(-1)
        
        pos_logits = logits[labels == 1]
        neg_logits = logits[labels == 0]
        
        pos_preds = torch.sigmoid(pos_logits.squeeze().float()).detach().cpu().numpy()
        neg_preds = torch.sigmoid(neg_logits.squeeze().float()).detach().cpu().numpy()
        preds.append(pos_preds)
        preds.append(neg_preds)
        
        targets = []
        pos_target = torch.ones(len(pos_preds))
        neg_target = torch.zeros(len(neg_preds))
        targets.append(pos_target)
        targets.append(neg_target)

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        return preds, targets, sample['target'].detach().cpu().numpy()
    
    @staticmethod
    def reduce_metrics(logging_outputs):

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_pos = sum(log.get("n_pos", 0) for log in logging_outputs)
        n_neg = sum(log.get("n_neg", 0) for log in logging_outputs)
        
        # pdb.set_trace()
        with torch.no_grad():
            logits = torch.cat([log.get("logits", 0) for log in logging_outputs])
        
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("pos_acc", 100.0 * pos_acc / n_pos, n_pos, round=1)
            neg_acc = sum(log.get("neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("neg_acc", 100.0 * neg_acc / n_neg, n_neg, round=1)

        if len(logging_outputs) > 0 and "inter_loss" in logging_outputs[0]:
            inter_loss_sum = sum(log.get("inter_loss", 0) for log in logging_outputs)
            metrics.log_scalar("inter_loss", inter_loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "intra_loss" in logging_outputs[0]:
            intra_loss_sum = sum(log.get("intra_loss", 0) for log in logging_outputs)
            metrics.log_scalar("intra_loss", intra_loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "t_ncorrect" in logging_outputs[0]:
            t_ncorrect = sum(log.get("t_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("t_accuracy", 100.0 * t_ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("t_pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("t_pos_acc", 100.0 * pos_acc / n_pos, n_pos, round=1)
            neg_acc = sum(log.get("t_neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("t_neg_acc", 100.0 * neg_acc / n_neg, n_neg, round=1)

        if len(logging_outputs) > 0 and "g_ncorrect" in logging_outputs[0]:
            g_ncorrect = sum(log.get("g_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("g_accuracy", 100.0 * g_ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("g_pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("g_pos_acc", 100.0 * pos_acc / n_pos, n_pos, round=1)
            neg_acc = sum(log.get("g_neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("g_neg_acc", 100.0 * neg_acc / n_neg, n_neg, round=1)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
