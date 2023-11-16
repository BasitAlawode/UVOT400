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
cfg.model.backbone.type = 'MAEEncode'
#
cfg.model.backbone.arch = 'base'
cfg.model.backbone.weights = './checkpoints/translate_template_common_pretrain/translate_template_common_pretrain_E500.pth'  # absolute path to pretrain checkpoint

# cfg.model.backbone.arch = 'small'
# cfg.model.backbone.weights = '/home/space/Documents/Experiments/BaseT/pretrain/vit-s16-moco3.pth'  # absolute path to pretrain checkpoint

#
cfg.model.backbone.lr_mult = 0.
cfg.model.backbone.train_layers = [11, 10, 9, 8, 7, 6]# [11, 10, 9] [11, 10, 9, 8, 7, 6] []
cfg.model.backbone.train_all = (cfg.model.backbone.lr_mult > 0) & (len(cfg.model.backbone.train_layers) == 0)


# # backbone
# cfg.model.backbone = Edict()
# cfg.model.backbone.search_size = cfg.model.search_size
# cfg.model.backbone.template_size = cfg.model.template_size
# cfg.model.backbone.type = 'ResNet'
# #
# cfg.model.backbone.arch = 'resnet50'
# cfg.model.backbone.norm_layer = None  # None for frozenBN
# cfg.model.backbone.use_pretrain = True
# cfg.model.backbone.dilation_list = [False, False, False]  # layer2 layer3 layer4, in increasing depth order
# #
# cfg.model.backbone.top_layer = 'layer3'
# cfg.model.backbone.use_inter_layer = False
# #
# cfg.model.backbone.lr_mult = 0.1
# cfg.model.backbone.train_layers = []
# cfg.model.backbone.train_all = (cfg.model.backbone.lr_mult > 0) & (len(cfg.model.backbone.train_layers) == 0)



# nlp
cfg.model.use_language = False

# neck
cfg.model.neck = Edict()
cfg.model.neck.search_size = []
cfg.model.neck.template_size = []
cfg.model.neck.in_channels_list = []
cfg.model.neck.inter_channels = 256
cfg.model.neck.type = 'DWCorr'
#
cfg.model.neck.transformer = Edict()
cfg.model.neck.transformer.in_channels = 256
cfg.model.neck.transformer.num_heads = 8
cfg.model.neck.transformer.dim_feed = 2048
cfg.model.neck.transformer.dropout = 0.1
cfg.model.neck.transformer.activation = 'relu'
cfg.model.neck.transformer.norm_before = False
cfg.model.neck.transformer.return_inter_decode = False
cfg.model.neck.transformer.num_encoders = 0
cfg.model.neck.transformer.num_decoders = 4

# head
cfg.model.head = Edict()
cfg.model.head.search_size = []
cfg.model.head.stride = -1
cfg.model.head.in_channels = cfg.model.neck.inter_channels
cfg.model.head.inter_channels = 256
cfg.model.head.type = 'Corner'

# criterion
cfg.model.criterion = Edict()
cfg.model.criterion.type = 'DETR'
#
cfg.model.criterion.alpha_giou = 2
cfg.model.criterion.alpha_l1 = 5
cfg.model.criterion.alpha_conf = 1

# ############################################
#                  data
# ############################################
cfg.data = Edict()
cfg.data.num_works = 8
cfg.data.batch_size = 32
cfg.data.sample_range = 200
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
cfg.trainer.end_epoch = 300
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
cfg.trainer.optim.weight_decay = 1e-4
#
cfg.trainer.optim.grad_clip_norm = None
cfg.trainer.optim.grad_acc_steps = 2
cfg.trainer.print_interval = cfg.trainer.optim.grad_acc_steps

# lr_scheduler
cfg.trainer.lr_scheduler = Edict()
cfg.trainer.lr_scheduler.type = 'multi_step'  # lr_scheduler list | 'cosine' 'multi_step'
cfg.trainer.lr_scheduler.warmup_epoch = 0
cfg.trainer.lr_scheduler.milestones = [240]

# tracker
cfg.tracker = Edict()
cfg.tracker.score_threshold = 0.0
cfg.tracker.name = f'translate_track'

if __name__ == '__main__':
    import pprint

    print('\n' + pprint.pformat(cfg))
