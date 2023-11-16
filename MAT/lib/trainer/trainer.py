import functools
import importlib
import logging
import os
import pprint
import sys
import time
from contextlib import nullcontext
from typing import Callable, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from register import path_register
from lib.trainer.lr_schedulers import lr_multi_step, lr_cosine
from utils.avgmeter import AverageMeter
from utils.misc import init_distributed_mode, get_rank, save_on_master


class Trainer(object):
    def __init__(self):
        self.path = None
        self.cfg = None

        self.logger = None

        self.writer_train = None
        self.writer_val = None

        self.model, self.model_without_ddp, self.nlp, self.nlp_without_ddp = None, None, None, None

        self.optimizer = None
        self.lr_scheduler = None

        self.loader_train, self.loader_val, self.sampler_train, self.sampler_val = None, None, None, None

    def init(self, **kwargs):
        args = kwargs.get('args', None)
        model_builder = kwargs.get('model_builder', None)
        dataset_fn: List[Callable] = kwargs.get('dataset_fn', None)
        find_unused: bool = kwargs.get('find_unused', None)

        self.path = path_register
        self.cfg = args

        # setting.distributed setting.gpu
        # don't call this function before `logging.basicConfig` if pytorch >= 1.8
        init_distributed_mode(self.cfg.trainer.dist)

        # prepare path
        self.prepare_path()

        # build logger
        self.logger = self.build_logger()

        # want reproducibility
        self.want_reproducibility()

        # build tensorboard
        self.writer_train = SummaryWriter(self.path.train_writer_path)
        self.writer_val = SummaryWriter(self.path.val_writer_path)

        # build model
        model_list = self.build_model(model_builder, find_unused)
        self.model, self.model_without_ddp, self.nlp, self.nlp_without_ddp = model_list

        self.load_pretrain()

        # build optimizer and lr_scheduler
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_lr_scheduler()

        # set train
        self.cfg.trainer.start_epoch = 0
        self.cfg.trainer.total_iter = 0
        self.cfg.trainer.current_epoch = self.cfg.trainer.start_epoch + 1

        # resume from checkpoint
        self.resume()

        # build data loader
        self.loader_train, self.loader_val, self.sampler_train, self.sampler_val = self.build_dataloader(dataset_fn)

        self.logger.debug('[>>> Initialized ! <<<]')

    def __call__(self, *args, **kwargs):
        # train loop
        self.logger.debug('[START training]')

        for i_epoch in range(self.cfg.trainer.start_epoch, self.cfg.trainer.end_epoch):
            self.cfg.trainer.current_epoch = i_epoch + 1

            self.writer_train.add_scalar(f'Metric/lr', self.lr_scheduler.get_last_lr()[-1], i_epoch)

            # train
            if self.cfg.trainer.dist.distributed:
                self.sampler_train.set_epoch(i_epoch)

            self.train_one_epoch()

            # save checkpoint
            if self.cfg.trainer.current_epoch % self.cfg.trainer.save_interval == 0:
                self.save_checkpoint()

            # validate
            if self.cfg.trainer.current_epoch % self.cfg.trainer.val_interval == 0 and self.loader_val is not None:

                if self.cfg.trainer.dist.distributed:
                    self.sampler_val.set_epoch(i_epoch)

                with torch.no_grad():
                    self.validate()

            # change learning rate
            self.lr_scheduler.step()

        # cleanup
        if self.cfg.trainer.dist.distributed:
            torch.distributed.destroy_process_group()

    def train_one_epoch(self):
        meter_dict = dict()

        grad_clip_norm = self.cfg.trainer.optim.grad_clip_norm
        grad_acc_steps = self.cfg.trainer.optim.grad_acc_steps

        print_interval = self.cfg.trainer.print_interval

        scaler = torch.cuda.amp.GradScaler()

        self.model.train()
        if self.cfg.model.use_language:
            self.nlp.train()

        # with torch.autograd.set_detect_anomaly(True):
        for i_iter, input_dict in enumerate(self.loader_train):
            tic = time.time()

            self.cfg.trainer.total_iter += 1

            input_dict.update({'training': True})

            if self.cfg.model.use_language:
                lang = input_dict.get('language', None)
                assert lang is not None, "not provide language"
            else:
                lang = None

            # no_sync(): A context manager to disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module variables,
            # which will later be synchronized in the first forward-backward pass exiting the context.
            if self.cfg.trainer.dist.distributed and self.cfg.trainer.total_iter % grad_acc_steps != 0:
                with_context = self.model.no_sync
            else:
                with_context = nullcontext

            if self.cfg.trainer.amp:  # auto mixed precision

                with with_context():

                    with torch.cuda.amp.autocast():

                        if self.cfg.model.use_language:
                            nlp_out, nlp_mask = self.nlp(lang)  # (N, L, 768) (N, L, 1)
                            input_dict.update({'language': nlp_out})
                            input_dict.update({'language_mask': nlp_mask})

                        loss, info_dict = self.model(input_dict)
                        loss = loss / grad_acc_steps

                    if not torch.isnan(loss):
                        scaler.scale(loss).backward()

                if self.cfg.trainer.total_iter % grad_acc_steps == 0:
                    if grad_clip_norm is not None:
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(self.optimizer)
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                        if self.cfg.model.use_language:
                            torch.nn.utils.clip_grad_norm_(self.nlp.parameters(), grad_clip_norm)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

            else:
                with with_context():

                    if self.cfg.model.use_language:
                        nlp_out, nlp_mask = self.nlp(lang)  # (N, L, 768) (N, L, 1)
                        input_dict.update({'language': nlp_out})
                        input_dict.update({'language_mask': nlp_mask})

                    loss, info_dict = self.model(input_dict)
                    loss = loss / grad_acc_steps
                    if not torch.isnan(loss):
                        loss.backward()

                if self.cfg.trainer.total_iter % grad_acc_steps == 0:
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                        if self.cfg.model.use_language:
                            torch.nn.utils.clip_grad_norm_(self.nlp.parameters(), grad_clip_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            toc = time.time()

            info_dict[0].update({'total': loss.item() * grad_acc_steps})
            info_dict[1].update({'ips': 1 / (toc - tic)})

            # logging
            self.logging_meter(meter_dict, info_dict)

            if self.cfg.trainer.total_iter % print_interval == 0:
                self.logging(i_iter, len(self.loader_train), info_dict)

                if self.cfg.trainer.total_iter % (print_interval * 500) == 0 and len(info_dict) == 3:
                    self.logging_tb(self.writer_train, meter_dict, info_dict)
                else:
                    self.logging_tb(self.writer_train, meter_dict, info_dict[:2])

            # if self.cfg.trainer.total_iter % (print_interval*100) == 0 and len(info_dict) > 2:

    def validate(self):
        meter_dict = dict()

        self.model.eval()
        if self.cfg.model.use_language:
            self.nlp.eval()

        for i_iter, input_dict in enumerate(self.loader_val):
            tic = time.time()

            input_dict.update({'training': True})

            if self.cfg.model.use_language:
                lang = input_dict.get('language', None)
                assert lang is not None, "not provide language"
            else:
                lang = None

            if self.cfg.trainer.amp:  # auto mixed precision
                with torch.cuda.amp.autocast():

                    if self.cfg.model.use_language:
                        nlp_out, nlp_mask = self.nlp(lang)  # (N, L, 768) (N, L, 1)
                        input_dict.update({'language': nlp_out})
                        input_dict.update({'language_mask': nlp_mask})

                    loss, info_dict = self.model(input_dict)

            else:
                if self.cfg.model.use_language:
                    nlp_out, nlp_mask = self.nlp(lang)  # (N, L, 768) (N, L, 1)
                    input_dict.update({'language': nlp_out})
                    input_dict.update({'language_mask': nlp_mask})

                loss, info_dict = self.model(input_dict)

            toc = time.time()

            info_dict[0].update({'total': loss.item()})
            info_dict[1].update({'ips': 1 / (toc - tic)})

            # logging
            self.logging_meter(meter_dict, info_dict[:2])
            self.logging(i_iter, len(self.loader_val), info_dict[:2], val=True)

        self.logging_tb(self.writer_val, meter_dict, info_dict)

    @staticmethod
    def logging_meter(meter_dict, info_dicts: List[dict]):
        for k, v in info_dicts[0].items():
            if k not in meter_dict:
                meter_dict.update({k: AverageMeter(wind_size=50)})
            meter_dict[k].update(v)

        for k, v in info_dicts[1].items():
            if k not in meter_dict:
                meter_dict.update({k: AverageMeter(wind_size=50)})
            meter_dict[k].update(v)

    def logging(self, i_iter, epoch_length, info_dicts: List[dict], val=False):
        start = "[{:>5d}/{:d}-{:3d}] ".format(i_iter + 1, epoch_length, self.cfg.trainer.current_epoch)
        loss = ' '.join(["{:s}: {:.4f}".format(k, v) for k, v in info_dicts[0].items()])
        metric = ' '.join(["{:s}: {:.2f}".format(k, v) for k, v in info_dicts[1].items()])

        if val:
            start = '>>>' + start
        self.logger.debug(' / '.join([start, loss, metric]))

    def logging_tb(self, writer, meter_dict, info_dicts: List[dict]):
        for k, v in info_dicts[0].items():
            writer.add_scalar(f'Loss/{k}', meter_dict[k].avg, self.cfg.trainer.total_iter)
        for k, v in info_dicts[1].items():
            writer.add_scalar(f'Metric/{k}', meter_dict[k].avg, self.cfg.trainer.total_iter)

        if len(info_dicts) > 2:
            for k, v in info_dicts[2].items():
                writer.add_images(f'Vis/{k}', v, self.cfg.trainer.total_iter)

    def prepare_path(self):
        os.makedirs(self.path.log_dir, exist_ok=True)
        os.makedirs(self.path.checkpoint_dir, exist_ok=True)

        self.path.current_log_dir = os.path.join(self.path.log_dir,
                                                 time.strftime('%Y-%m-%d--%H-%M_{}'.format(self.cfg.exp_name)))
        os.makedirs(self.path.current_log_dir, exist_ok=True)

        self.path.train_writer_path = os.path.join(self.path.current_log_dir,
                                                   'train_{}'.format(self.cfg.trainer.dist.local_rank))
        os.makedirs(self.path.train_writer_path, exist_ok=True)

        self.path.val_writer_path = os.path.join(self.path.current_log_dir,
                                                 'val_{}'.format(self.cfg.trainer.dist.local_rank))
        os.makedirs(self.path.val_writer_path, exist_ok=True)

    @functools.lru_cache()
    def build_logger(self):
        # create logger
        logger = logging.getLogger('LocalRank {}'.format(self.cfg.trainer.dist.local_rank))
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # create formatter
        fmt = '[%(asctime)s] -- %(levelname)s: %(message)s'

        # create console handlers for master process
        if self.cfg.trainer.dist.local_rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(console_handler)

        # create file handlers
        file_handler = logging.FileHandler(os.path.join(self.path.current_log_dir, 'train.log'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # INFO > DEBUG
        logger.setLevel(logging.DEBUG) if self.cfg.trainer.dist.local_rank == 0 else logger.setLevel(logging.INFO)

        # print settings
        logger.debug('[Check config]')
        logger.debug('\n' + pprint.pformat(self.cfg))

        logger.info('[LocalRank:{}--DDP:{}, world_size:{}, local_rank:{}, master://{}:{}]'.format(
            self.cfg.trainer.dist.local_rank, self.cfg.trainer.dist.distributed, self.cfg.trainer.dist.world_size,
            self.cfg.trainer.dist.local_rank, self.cfg.trainer.dist.master_addr, self.cfg.trainer.dist.master_port))

        return logger

    def want_reproducibility(self):
        seed = self.cfg.trainer.seed + get_rank()
        self.logger.info('[LocalRank:{}, set seed {}]'.format(self.cfg.trainer.dist.local_rank, seed))

        import numpy as np
        np.random.seed(seed)
        import random
        random.seed(seed)

        torch.manual_seed(seed)

        # causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance
        torch.backends.cudnn.benchmark = False

        if getattr(self.cfg.trainer, 'deterministic', True):
            # While disabling CUDA convolution benchmarking (discussed above) ensures that CUDA selects the same
            # algorithm each time an application is run, that algorithm itself may be nondeterministic, unless either
            # torch.use_deterministic_algorithms(True) or torch.backends.cudnn.deterministic = True is set.
            # The latter setting controls only this behavior, unlike torch.use_deterministic_algorithms()
            # which will make other PyTorch operations behave deterministically, too.
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  # <-- for cuda>=10.2
            if int(torch.__version__.split('.')[0]) == 1 and int(torch.__version__.split('.')[1]) >= 8:  # pytorch>=1.8
                torch.use_deterministic_algorithms(True)
            else:
                torch.backends.cudnn.deterministic = True

    def build_model(self, model_builder, find_unused):
        self.logger.debug('[Build model]')

        model = model_builder(self.cfg.model).cuda()
        model_without_ddp = model

        if self.cfg.model.use_language:
            nlp_module = importlib.import_module('lib.model.nlp_models')
            if self.cfg.model.nlp_model.type == 'CLIP':
                nlp = getattr(nlp_module, self.cfg.model.nlp_model.type)(
                    self.cfg.model.nlp_model.lr_mult, self.cfg.model.nlp_model.arch).cuda()
            else:
                nlp = getattr(nlp_module, self.cfg.model.nlp_model.type)(self.cfg.model.nlp_model.lr_mult).cuda()

            # nlp = BERT().cuda()

            nlp_without_ddp = nlp
        else:
            nlp = None
            nlp_without_ddp = None

        if self.cfg.trainer.sync_bn:
            self.logger.debug('[Use syncBN]')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if self.cfg.model.use_language:
                nlp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nlp)

        if self.cfg.trainer.dist.distributed:
            self.logger.debug('[Use DDP]')
            model = DDP(model, device_ids=[self.cfg.trainer.dist.local_rank], find_unused_parameters=find_unused)
            model_without_ddp = model.module
            if self.cfg.model.use_language and self.cfg.model.nlp_model.lr_mult > 0:
                nlp = DDP(nlp, device_ids=[self.cfg.trainer.dist.local_rank], find_unused_parameters=find_unused)
                nlp_without_ddp = nlp.module

        return [model, model_without_ddp, nlp, nlp_without_ddp]

    def build_dataloader(self, dataset_fn):
        self.logger.debug('[Build data_loader]')

        val_flag = (len(self.cfg.data.datasets_train) > 0)

        dataset_build_fn, dataset_collate_fn = dataset_fn
        train_set, val_set = dataset_build_fn(self.cfg.data, lmdb=self.path.lmdb, json=self.path.json)

        if self.cfg.trainer.dist.distributed:
            sampler_train = DistributedSampler(train_set, shuffle=True)
            sampler_val = DistributedSampler(val_set, shuffle=True)
        else:
            sampler_train = RandomSampler(train_set)
            sampler_val = RandomSampler(val_set)

        batch_sampler_train = BatchSampler(sampler_train, batch_size=self.cfg.data.batch_size, drop_last=True)
        batch_sampler_val = BatchSampler(sampler_val, batch_size=max(1, self.cfg.data.batch_size // 4), drop_last=True)

        loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                  num_workers=self.cfg.data.num_works,
                                  collate_fn=dataset_collate_fn)

        loader_val = DataLoader(val_set, batch_sampler=batch_sampler_val,
                                num_workers=self.cfg.data.num_works,
                                collate_fn=dataset_collate_fn) if val_flag else None

        return loader_train, loader_val, sampler_train, sampler_val

    def build_optimizer(self):
        self.logger.debug('[Build optimizer]')

        model = self.model_without_ddp
        nlp = self.nlp_without_ddp
        pretrain_name = self.model_without_ddp.pretrained_param

        base_lr = self.cfg.trainer.optim.base_lr
        backbone_lr_mult = self.cfg.model.backbone.lr_mult
        pretrain_lr_mult = self.cfg.trainer.pretrain_lr_mult

        weight_decay = self.cfg.trainer.optim.weight_decay
        momentum = self.cfg.trainer.optim.momentum
        optim_type = self.cfg.trainer.optim.type

        if pretrain_name is not None:
            if self.cfg.trainer.pretrain_exclude is not None:
                exclude_name = [n for n in pretrain_name if self.cfg.trainer.pretrain_exclude in n]
                pretrain_name = [n for n in pretrain_name if self.cfg.trainer.pretrain_exclude not in n]

                self.logger.debug('[Check learnable pretrained parameters]')
                self.logger.debug('\n' + pprint.pformat(pretrain_name))
                self.logger.debug('[Check excluded pretrained parameters]')
                self.logger.debug('\n' + pprint.pformat(exclude_name))

            param_dicts = []
            if len([p for n, p in model.named_parameters() if p.requires_grad and n not in pretrain_name]) > 0:
                param_dicts += [
                    {"params": [p for n, p in model.named_parameters()
                                if p.requires_grad and n not in pretrain_name],
                     'lr': base_lr},
                ]

            if pretrain_lr_mult > 0:
                param_dicts += [
                    {"params": [p for n, p in model.named_parameters()
                                if 'backbone' in n and p.requires_grad and n in pretrain_name],
                     'lr': base_lr * backbone_lr_mult * pretrain_lr_mult},

                    {"params": [p for n, p in model.named_parameters()
                                if 'backbone' not in n and p.requires_grad and n in pretrain_name],
                     'lr': base_lr * pretrain_lr_mult},
                ]

            if getattr(self.cfg.model, 'use_language', False):
                nlp_lr_mult = self.cfg.model.nlp_model.lr_mult
                if nlp_lr_mult > 0:
                    assert 'bert' in pretrain_name, "not find language model in pre-trained model"
                    param_dicts = param_dicts + [
                        {"params": [p for n, p in nlp.named_parameters()
                                    if 'bert' in n and p.requires_grad and n in pretrain_name],
                         'lr': base_lr * pretrain_lr_mult * nlp_lr_mult},
                    ]
                else:
                    for name, parameter in model.named_parameters():
                        if 'bert' in pretrain_name:
                            parameter.requires_grad_(False)

        else:
            self.logger.debug("[No pretrained params]")

            param_dicts = [
                {"params": [p for n, p in model.named_parameters()
                            if 'backbone' in n and p.requires_grad],
                 'lr': base_lr * backbone_lr_mult},

                {"params": [p for n, p in model.named_parameters()
                            if 'backbone' not in n and p.requires_grad],
                 'lr': base_lr},
            ]

            if getattr(self.cfg.model, 'use_language', False):
                nlp_lr_mult = self.cfg.model.nlp_model.lr_mult
                param_dicts = param_dicts + [
                    {"params": [p for n, p in nlp.named_parameters() if p.requires_grad],
                     'lr': base_lr * nlp_lr_mult},
                ]

        # print learnable parameters
        self.logger.debug('[Check learnable parameters]')
        param_list = [n for n, p in model.named_parameters() if p.requires_grad]
        for n in param_list.copy():
            if 'backbone' in n and backbone_lr_mult == 0:
                param_list.remove(n)

            if pretrain_name is not None:
                if n in pretrain_name and pretrain_lr_mult == 0:
                    param_list.remove(n)

        if self.cfg.model.use_language:
            param_list = param_list + [n for n, p in self.nlp_without_ddp.named_parameters() if p.requires_grad]

        self.logger.debug('\n' + pprint.pformat(param_list))
        self.logger.debug('[Learnable params: {} M]'.format(
            sum(p.numel() for n, p in model.named_parameters() if n in param_list) / 1e6))

        if optim_type == 'SGD':
            optimizer = SGD(param_dicts, lr=base_lr,
                            weight_decay=weight_decay,
                            momentum=momentum)

        elif optim_type == 'AdamW':
            optimizer = AdamW(param_dicts, lr=base_lr,
                              weight_decay=weight_decay)

        else:
            raise NotImplementedError("{}: not implemented!".format(optim_type))

        return optimizer

    def build_lr_scheduler(self):
        self.logger.debug('[Build lr_scheduler]')

        total_epochs = self.cfg.trainer.end_epoch
        scheduler_type = self.cfg.trainer.lr_scheduler.type
        warmup_epoch = self.cfg.trainer.lr_scheduler.warmup_epoch
        milestones = self.cfg.trainer.lr_scheduler.milestones

        if scheduler_type == 'multi_step':
            lr_lambda = functools.partial(lr_multi_step,
                                          warmup_epoch=warmup_epoch,
                                          milestones=milestones)

        elif scheduler_type == 'cosine':
            lr_lambda = functools.partial(lr_cosine,
                                          warmup_epoch=warmup_epoch,
                                          total_epochs=total_epochs)

        else:
            raise NotImplementedError("{}: not implemented!".format(scheduler_type))

        lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        return lr_scheduler

    def load_pretrain(self):
        if self.cfg.trainer.pretrain is not None:
            checkpoint = torch.load(os.path.join(self.path.pretrain_dir, self.cfg.trainer.pretrain), map_location='cpu')

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
                model_dict = self.model_without_ddp.state_dict()

                # filter out unnecessary keys
                filter_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

                self.model_without_ddp.pretrained_param = list(filter_pretrained_dict.keys())
                unused_param = [k for k, v in pretrained_dict.items() if k not in model_dict]
                lost_param = [k for k, v in model_dict.items() if k not in pretrained_dict]

                # overwrite entries in the existing state dict & load the new state dict
                model_dict.update(filter_pretrained_dict)
                self.model_without_ddp.load_state_dict(model_dict)

                self.logger.debug('[Load pretrain from: {}]'.format(self.cfg.trainer.pretrain))
                self.logger.debug('----unused param:' + '\n' + pprint.pformat(unused_param))
                self.logger.debug('----lost_param:' + '\n' + pprint.pformat(lost_param))

    def resume(self):
        if self.cfg.trainer.resume is not None:
            checkpoint = torch.load(os.path.join(self.path.checkpoint_dir, self.cfg.exp_name,
                                                 '{}_E{:0>3d}.pth'.format(self.cfg.exp_name, self.cfg.trainer.resume)),
                                    map_location='cpu')

            self.cfg.trainer.start_epoch = checkpoint['epoch']
            self.cfg.trainer.total_iter = checkpoint['total_iter']

            self.model_without_ddp.load_state_dict(checkpoint['model'])

            if self.cfg.model.use_language:
                assert 'nlp_model' in checkpoint, "not find language model"
                self.nlp_without_ddp.load_state_dict(checkpoint['nlp_model'])

            assert 'optimizer' in checkpoint, "not find optimizer"
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            assert 'lr_scheduler' in checkpoint, "not find lr_scheduler"
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            self.logger.debug('[RESUME from {}, lr {}]'.format(
                self.cfg.trainer.resume, self.lr_scheduler.get_last_lr()))

    def save_checkpoint(self):
        os.makedirs(os.path.join(self.path.checkpoint_dir, self.cfg.exp_name), exist_ok=True)
        checkpoint_path = os.path.join(self.path.checkpoint_dir, self.cfg.exp_name,
                                       '{}_E{:0>3d}.pth'.format(self.cfg.exp_name, self.cfg.trainer.current_epoch))

        save_dict = {
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.cfg.trainer.current_epoch,
            'total_iter': self.cfg.trainer.total_iter,
            'cfg': self.cfg,
        }

        if self.cfg.model.use_language:
            save_dict.update({'nlp_model': self.nlp_without_ddp.state_dict()})

        save_on_master(save_dict, checkpoint_path)

        if self.cfg.trainer.dist.distributed:
            # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
            dist.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.cfg.trainer.dist.local_rank}

            ckp = torch.load(checkpoint_path, map_location=map_location)
            self.model_without_ddp.load_state_dict(ckp['model'])
            if self.cfg.model.use_language:
                self.nlp_without_ddp.load_state_dict(ckp['nlp_model'])

            # Use a barrier() to make sure that all processes have finished reading the checkpoint
            dist.barrier()
