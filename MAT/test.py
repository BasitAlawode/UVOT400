import os
import sys
import argparse
import copy
from os.path import join as p_join
from easydict import EasyDict as Edict

sys.path.append('/')
from register import path_register as path
from register import exp_register, benchmark_register, data_register
from lib.tester import Tester
try:
    from lib.tester import TesterTRT
except ImportError:
    print("!!! >>> need TensorRT for 'from lib.tester import TesterTRT' <<< !!!")


def eval_one_model(cfg):

    gpu_ids = cfg.gpu_id
    cfg.num_gpu = len(cfg.gpu_id.strip().replace(',', ''))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

    if ('got10k' in cfg.benchmark or 'lasot' in cfg.benchmark
            or 'nfs' in cfg.benchmark or 'otb' in cfg.benchmark or 'uav123' in cfg.benchmark
            or 'trackingnet' in cfg.benchmark or 'vot2019lt' in cfg.benchmark or 'vot2018lt' in cfg.benchmark
            or 'tnl2k' in cfg.benchmark or 'vid_sentence' in cfg.benchmark or 'otb99lang' in cfg.benchmark
            or 'trek150' in cfg.benchmark or 'itb' in cfg.benchmark or 'vot2022lt' in cfg.benchmark):

        exp_args: dict = copy.deepcopy(exp_register[cfg.experiment])
        exp_args['args'].exp_name = '{}_{}'.format(cfg.experiment, cfg.train_set)

        tracker_class = exp_args['tracker']
        exp_args.update({'tracker': Edict()})
        exp_args['tracker'].update(exp_args['args'].tracker)

        exp_args['tracker'].tracker_class = tracker_class
        exp_args['tracker'].name = '{}_E{:0>3d}'.format(exp_args['args'].tracker.name, cfg.test_epoch)

        if cfg.test_epoch != -1:
            exp_args['tracker'].ckp_path = p_join(path.checkpoint_dir, exp_args['args'].exp_name,
                                                  '{}_E{:0>3d}.pth'.format(exp_args['args'].exp_name, cfg.test_epoch))
        else:
            exp_args['tracker'].ckp_path = None

        if cfg.trt:
            exp_args['tracker'].trt_path = p_join(path.checkpoint_dir+'_trt', exp_args['args'].exp_name,
                                                  exp_args['tracker'].name + '_dynamic.trt')

        exp_args['tracker'].hyper = cfg.hyper
        if exp_args['tracker'].hyper is not None:
            exp_args['tracker'].name += '_' + str(exp_args['tracker'].hyper)
        exp_args['tracker'].longterm = cfg.longterm
        exp_args['tracker'].vis = cfg.vis
        exp_args['tracker'].fp16 = cfg.fp16
        exp_args['tracker'].visualize = False  # <--- save attention map
        exp_args['tracker'].template_sf = exp_args['args'].data.template_scale_f
        exp_args['tracker'].template_sz = exp_args['args'].data.template_size
        exp_args['tracker'].search_sf = exp_args['args'].data.search_scale_f
        exp_args['tracker'].search_sz = exp_args['args'].data.search_size

        exp_args.update({'tester': Edict()})
        exp_args['tester'].benchmark = cfg.benchmark
        exp_args['tester'].num_process = cfg.num_process
        exp_args['tester'].num_gpu = cfg.num_gpu
        exp_args['tester'].res_dir = p_join(path.result_dir, cfg.benchmark, exp_args['args'].exp_name,
                                            exp_args['tracker'].name)

        os.makedirs(exp_args['tester'].res_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_args['tester'].res_dir, 'times'), exist_ok=True)
        os.makedirs(os.path.join(exp_args['tester'].res_dir, 'scores'), exist_ok=True)

        print('test tracker: <{}> using model <{}> on benchmark <{}> with <{:d}> processes on <{}> gpus'.format(
            exp_args['tracker'].name, exp_args['tracker'].ckp_path,
            exp_args['tester'].benchmark, exp_args['tester'].num_process, gpu_ids))

        if cfg.trt:
            print('+++++ Use TensorRT +++++')
            tester = TesterTRT(**exp_args)
        else:
            tester = Tester(**exp_args)
        tester.run_ope()
    else:
        raise NotImplementedError(f'"{cfg.benchmark}" benchmark is not implemented')


if __name__ == '__main__':
    # ################### pool OSError: [Errno 24] Too many open files ###################
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024*8, rlimit[1]))
    # #################################################################################

    parser = argparse.ArgumentParser(description='multi-gpu test')

    parser.add_argument('-e', '--experiment', default='translate_track',
                        type=str, required=False,
                        choices=list(exp_register.keys()),
                        help='the name of experiment -- check `./lib/register/experiments.py` to get more '
                             'information about each experiment.')

    parser.add_argument('-t', '--train_set', default='common',
                        type=str, required=False,
                        choices=list(data_register.keys()),
                        help='the name of train set -- check `./lib/register/datasets.py` to get more information '
                             'about each train set.')

    parser.add_argument('-b', '--benchmark', default='lasot',
                        type=str, required=False,
                        choices=benchmark_register['choices'],
                        help='the name of benchmark -- check `./lib/register/benchmarks.py` to get more information '
                             'about each benchmark.'
                        )

    parser.add_argument('--test_epoch', default=300,
                        type=int, required=False,
                        help='test which epoch -- for example, `100` indicates we load `checkpoint_100.pth` '
                             'and test.')

    parser.add_argument('--num_process', default=0,
                        type=int, required=False,
                        help='max processes each time, set 0 for single-process test.')

    parser.add_argument('--gpu_id', default='0,1',
                        type=str, required=False,
                        help='CUDA_VISIBLE_DEVICES')

    parser.add_argument('--longterm', dest='longterm', action='store_true', default=False,
                        help='tracking in long-term mode.')

    parser.add_argument('--vis', dest='vis', action='store_true', default=False,
                        help='show tracking result.')

    parser.add_argument('--fp16', dest='fp16', action='store_true', default=False,
                        help='tracking with float16.')

    parser.add_argument('--trt', dest='trt', action='store_true', default=False,
                        help='tracking with TensorRT.')

    args = parser.parse_args()
    args.hyper = None

    if args.benchmark == 'vot2019lt':
        assert args.longterm is True, 'please set `longterm` Flag for long-term tracking'

    eval_one_model(args)
