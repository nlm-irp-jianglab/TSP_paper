import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import pickle
from collections import defaultdict

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss, metric_ftns, optimizer, config, device,
                 data_loader, batch_transform, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        # len_epoch is used to set how many times we should iterate the batches in an epoch, we can stop early or keep looping the datasets
        if len_epoch is None:
            # epoch-based training
            # the length of data_loader = number of batches = dataset_size / batch_size
            # e.g. we have 216465 samples and batch size=512, so the batch number=216465/512~=423
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
            
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.batch_transform = batch_transform

        self.train_metrics = MetricTracker('loss', writer=self.writer) # only train loss can be updated per batch
        self.valid_metrics = MetricTracker('loss', writer=self.writer) # only valid loss can be updated per batch
        # other batch need to be updated per epoch
        self.train_epoch_metrics = MetricTracker('total_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_epoch_metrics = MetricTracker('total_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.train_epoch_metrics.reset()

        total_output = torch.tensor([]).to(self.device)
        total_target = torch.tensor([]).to(self.device).long()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # deal with each batch
            # data: a batch of sequence with each res to a number defined by ESM,
            # padding is added to make each sequence equal lengths (max_seq_len) and extra 2 lables <cls> and <seg> are added, 
            # shape=(batch size, max_seq_len+2)
            # target: a batch of the true binary labels, shape=(batch size, 1)
            # mask: a batch of valid lengths for sequences, shape=(batch size, 1)
            mask = data[1]
            if self.batch_transform:
                data = self.batch_transform(data[0])
            else:
                data = data[0]
            
            # print(data)
            # print(data.shape)
            # print(target)
            # print(target.shape)

            # move data to GPU device
            data, target, mask = data.to(self.device), target.to(self.device).long(), mask.to(self.device)

            self.optimizer.zero_grad()
            # train model with each batch
            output = self.model(data, mask) # shape=(batch_size, 2)
            output = output.to(self.device)
            
            total_output = torch.cat((total_output, output), 0)
            total_target = torch.cat((total_target, target), 0)
            
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item()) # only loss
            
            # printout each batch train loss
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        # add total metrics to log
        loss = self.loss(total_output, total_target)
        self.train_epoch_metrics.update("total_loss", loss.item())
        
        for met in self.metric_ftns:
            metric = met(total_output, total_target)
            self.train_epoch_metrics.update(met.__name__, metric)
        
        # Note that this train metric is only for this epoch
        # since it onlu update instead of accumulating
        log = self.train_metrics.result()
        total_log = self.train_epoch_metrics.result()
        
        # add total loss to log
        log.update(**{k : v for k, v in total_log.items()})
        
        if self.do_validation:
            val_log, total_val_log = self._valid_epoch(epoch)
            # val_loss is updated here and added to the log object
            # Note that this validation metric is only for this epoch
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{'val_'+k : v for k, v in total_val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval() # do not train the model
        self.valid_metrics.reset()
        self.valid_epoch_metrics.reset()
        
        total_output = torch.tensor([]).to(self.device)
        total_target = torch.tensor([]).to(self.device).long()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                mask = data[1]
                if self.batch_transform:
                    data = self.batch_transform(data[0])
                else:
                    data = data[0]

                data, target, mask = data.to(self.device), target.to(self.device).long(), mask.to(self.device)
                output = self.model(data, mask)
                output = output.to(self.device)
                
                total_output = torch.cat((total_output, output), 0)
                total_target = torch.cat((total_target, target), 0)
            
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item()) # only loss

        # add total metrics to log
        loss = self.loss(total_output, total_target)
        self.valid_epoch_metrics.update("total_loss", loss.item())
        
        for met in self.metric_ftns:
            metric = met(total_output, total_target)
            self.valid_epoch_metrics.update(met.__name__, metric)
        
        val_log = self.valid_metrics.result()
        total_val_log = self.valid_epoch_metrics.result()
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        
        return val_log, total_val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
