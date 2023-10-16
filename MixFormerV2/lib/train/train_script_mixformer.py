import os
# loss function related
from lib.utils.box_ops import ciou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mixformer2_vit import build_mixformer2_vit, build_mixformer2_vit_online
# forward propagation related
from lib.train.actors import MixFormerActor
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for Mixformer'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == "mixformer2_vit":
        net = build_mixformer2_vit(cfg, settings)
    elif settings.script_name == "mixformer2_vit_online":
        net = build_mixformer2_vit_online(cfg, settings)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # settings.save_every_epoch = True
    # Loss functions and Actors
    if settings.script_name == 'mixformer':
        objective = {'ciou': ciou_loss, 'l1': l1_loss}
        loss_weight = {'ciou': cfg.TRAIN.ciou_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == 'mixformer_online':
        objective = {'ciou': ciou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss()}
        loss_weight = {'ciou': cfg.TRAIN.ciou_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)
    elif settings.script_name == 'mixformer2_vit':
        objective = {'ciou': ciou_loss, 'l1': l1_loss}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == 'mixformer2_vit_online':
        objective = {'ciou': ciou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss()}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
