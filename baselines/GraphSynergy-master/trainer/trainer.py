import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 criterion, 
                 metric_fns, 
                 optimizer, 
                 config, 
                 data_loader,
                 feature_index, 
                 cell_neighbor_set, 
                 drug_neighbor_set, 
                 valid_data_loader=None, 
                 test_data_loader=None, 
                 lr_scheduler=None, 
                 len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        # for data
        self.data_loader = data_loader
        self.cell_neighbor_set = cell_neighbor_set
        self.drug_neighbor_set = drug_neighbor_set
        self.feature_index = feature_index

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            target = target.to(self.device)
            output, emb_loss = self.model(*self._get_feed_dict(data))
            loss = self.criterion(output, target.squeeze()) + emb_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.train_metrics.update(met.__name__, met(y_pred, y_true))
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_'+k : v for k, v in val_log.items()}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                target = target.to(self.device)
                output, emb_loss = self.model(*self._get_feed_dict(data))
                loss = self.criterion(output, target.squeeze()) + emb_loss

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(y_pred, y_true))
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self):
        self.model.eval()
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                target = target.to(self.device)
                output, emb_loss = self.model(*self._get_feed_dict(data))
                loss = self.criterion(output, target.squeeze()) + emb_loss

                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for i, metric in enumerate(self.metric_fns):
                    total_metrics[i] += metric(y_pred, y_true) * batch_size
        
        test_output = {'n_samples': len(self.test_data_loader.sampler), 
                       'total_loss': total_loss, 
                       'total_metrics': total_metrics}
        
        return test_output

    def get_save(self, save_files):
        result = dict()
        for key, value in save_files.items():
            if type(value) == dict:
                temp = dict()
                for k,v in value.items():
                    temp[k] = v.cpu().detach().numpy()
            else:
                temp = value.cpu().detach().numpy()
            result[key] = temp
        return result

    def _get_feed_dict(self, data):
        # [batch_size]
        cells = data[:, self.feature_index['cell']]
        drugs1 = data[:, self.feature_index['drug1']]
        drugs2 = data[:, self.feature_index['drug2']]
        cells_neighbors, drugs1_neighbors, drugs2_neighbors = [], [], []
        for hop in range(self.model.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.numpy()]).to(self.device))
            drugs1_neighbors.append(torch.LongTensor([self.drug_neighbor_set[d][hop] \
                                                          for d in drugs1.numpy()]).to(self.device))
            drugs2_neighbors.append(torch.LongTensor([self.drug_neighbor_set[d][hop] \
                                                          for d in drugs2.numpy()]).to(self.device))
        
        return cells.to(self.device), drugs1.to(self.device), drugs2.to(self.device), \
               cells_neighbors, drugs1_neighbors, drugs2_neighbors

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
