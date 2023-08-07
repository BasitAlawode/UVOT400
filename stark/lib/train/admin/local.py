class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/basit/Documents/PhD_Research/Codes/Tracking/pretrained_networks'
        self.lasot_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/lasot'
        self.got10k_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/got10k'
        self.lasot_lmdb_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/got10k_lmdb'
        self.trackingnet_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/trackingnet_lmdb'
        self.coco_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/coco'
        self.coco_lmdb_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/vid'
        self.imagenet_lmdb_dir = '/home/basit/Documents/PhD_Research/Codes/Tracking/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
