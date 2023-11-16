from easydict import EasyDict as Edict

cfg = Edict()

# ############################################
#                  model
# ############################################
cfg.model = Edict()
cfg.model.search_size = [224, 224]  # h, w
cfg.model.template_size = [112, 112]

# backbone
cfg.model.backbone = Edict()
cfg.model.backbone.search_size = cfg.model.search_size
cfg.model.backbone.template_size = cfg.model.template_size
cfg.model.backbone.type = 'MAE'
#
cfg.model.backbone.arch = 'base'
cfg.model.backbone.weights = None  # absolute path to pretrain checkpoint

# cfg.model.backbone.arch = 'small'
# cfg.model.backbone.weights = '/home/space/Documents/Experiments/BaseT/pretrain/vit-s16-moco3.pth'  # absolute path to pretrain checkpoint

#
cfg.model.backbone.lr_mult = 1
cfg.model.backbone.train_layers = []
cfg.model.backbone.train_all = (cfg.model.backbone.lr_mult > 0) & (len(cfg.model.backbone.train_layers) == 0)
cfg.model.backbone.from_scratch = False
# nlp
cfg.model.use_language = False


# ############################################
#                  data
# ############################################
cfg.data = Edict()
cfg.data.num_works = 8
cfg.data.batch_size = 32
cfg.data.sample_range = 9999
#
cfg.data.datasets_train = []
cfg.data.datasets_val = []
#
cfg.data.num_samples_train = 64000
cfg.data.num_samples_val = 1000
#
cfg.data.search_size = cfg.model.search_size
cfg.data.search_scale_f = 4.0
cfg.data.search_jitter_f = [0.5, 3]
#
cfg.data.template_size = cfg.model.template_size
cfg.data.template_scale_f = 2
cfg.data.template_jitter_f = [0.0, 0.0]

# ############################################
#                  trainer
# ############################################
cfg.trainer = Edict()
cfg.trainer.deterministic = False
cfg.trainer.seed = 123
cfg.trainer.print_interval = None
cfg.trainer.start_epoch = 0
cfg.trainer.end_epoch = 500
cfg.trainer.sync_bn = False
cfg.trainer.amp = True
#
cfg.trainer.resume = None
cfg.trainer.pretrain = None
cfg.trainer.pretrain_lr_mult = None
#
cfg.trainer.val_interval = 5
cfg.trainer.save_interval = 1

# distributed train
cfg.trainer.dist = Edict()
cfg.trainer.dist.distributed = False
cfg.trainer.dist.master_addr = None
cfg.trainer.dist.master_port = None
#
cfg.trainer.dist.device = 'cuda'
cfg.trainer.dist.world_size = None
cfg.trainer.dist.local_rank = None
cfg.trainer.dist.rank = None

# optimizer
cfg.trainer.optim = Edict()
cfg.trainer.optim.type = 'AdamW'
#
cfg.trainer.optim.base_lr = 1e-4
cfg.trainer.optim.momentum = 0.9
cfg.trainer.optim.weight_decay = 5e-2
#
cfg.trainer.optim.grad_clip_norm = None
cfg.trainer.optim.grad_acc_steps = 2
cfg.trainer.print_interval = cfg.trainer.optim.grad_acc_steps

# lr_scheduler
cfg.trainer.lr_scheduler = Edict()
cfg.trainer.lr_scheduler.type = 'multi_step'  # lr_scheduler list | 'cosine' 'multi_step'
cfg.trainer.lr_scheduler.warmup_epoch = 0
cfg.trainer.lr_scheduler.milestones = [200]

if __name__ == '__main__':
    import pprint

    print('\n' + pprint.pformat(cfg))
