import os
import sys
import argparse

sys.path.append('/')
from register import exp_register, data_register
from lib.trainer import Trainer

if __name__ == '__main__':
    # ################### RuntimeError: received 0 items of ancdata ###################
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024 * 8, rlimit[1]))
    # #################################################################################

    parser = argparse.ArgumentParser(description='multi-gpu train')
    parser.add_argument('--local_rank', type=int)

    parser.add_argument('-e', '--experiment', default='translate_deit_track',
                        type=str, required=False,
                        choices=list(exp_register.keys()),
                        help='the name of experiment -- check `./lib/register/experiments.py` to get more '
                             'information about each experiment.')

    parser.add_argument('-t', '--train_set', default='got10k',
                        type=str, required=False,
                        choices=list(data_register.keys()),
                        help='the name of train set -- check `./lib/register/datasets.py` to get more information '
                             'about each train set.')

    parser.add_argument('--resume_epoch', default=None,
                        type=int, required=False,
                        help='resume from which epoch -- for example, `100` indicates we load `checkpoint_100.pth` '
                             'and resume training.')

    parser.add_argument('--pretrain_name', default=None,
                        type=str, required=False,
                        help='the full name of the pre-trained model file -- for example, `checkpoint_100.pth` '
                             'indicates we load `./pretrain/checkpoint_100.pth`.')

    parser.add_argument('--pretrain_lr_mult', default=None,
                        type=float, required=False,
                        help='pretrain_lr = pretrain_lr_mult * base_lr -- load pre-trained weights and fine tune '
                             'these parameters with pretrain_lr.')

    parser.add_argument('--pretrain_exclude', default=None,
                        type=str, required=False,
                        help='the keyword of the name of pre-trained parameters that we want to discard -- for '
                             'example, `head` indicates we do not load the parameters whose name contains `head`.')

    parser.add_argument('--gpu_id', default='0,1',
                        type=str, required=False,
                        help='CUDA_VISIBLE_DEVICES')

    parser.add_argument('--find_unused', dest='find_unused', action='store_true', default=False,
                        help='used in DDP mode')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    train_val_set: list = data_register[args.train_set]

    exp_args: dict = exp_register[args.experiment]
    exp_args['args'].exp_name = '{}_{}'.format(args.experiment, args.train_set)
    exp_args['args'].data.datasets_train = train_val_set[0]
    exp_args['args'].data.datasets_val = train_val_set[1]

    exp_args['args'].trainer.resume = args.resume_epoch

    exp_args['args'].trainer.pretrain_exclude = args.pretrain_exclude
    if args.pretrain_name is not None:
        assert exp_args['args'].trainer.pretrain is None, "pretrained CKP has been set in config"
        exp_args['args'].trainer.pretrain = args.pretrain_name

        if args.pretrain_lr_mult is not None:
            assert exp_args['args'].trainer.pretrain_lr_mult is None, "pretrain lr_mult has been set in config"
            exp_args['args'].trainer.pretrain_lr_mult = args.pretrain_lr_mult

    exp_args.update({'find_unused': args.find_unused})

    trainer = Trainer()
    trainer.init(**exp_args)

    trainer()
    # python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0
    # --master_addr=10.7.9.26 --master_port=1234 train.py

    # CUDA_LAUNCH_BLOCKING=1
    # python -m torch.distributed.launch --nproc_per_node=2  train.py --experiment=dev --train_set=language --find_unused

    # kill $(ps aux | grep 'train' | grep -v grep | awk '{print $2}')
