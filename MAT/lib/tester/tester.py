import importlib
import multiprocessing as multiprocessing
import os
import time

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert, box_iou

from MAT.register import benchmark_register as benchmark_loader
from MAT.register import evaluator_register as benchmark_evaluator
from MAT.register import path_register as paths


def load_ckp(ckp_path, model, nlp_model=None):
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    map_location = torch.device('cpu')
    ckp = torch.load(ckp_path, map_location=map_location)

    model_dict = model.state_dict()
    ckp_dict = ckp['model']

    pretrained_dict = {k: v for k, v in ckp_dict.items() if k in model_dict}
    unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
    lost_param = [k for k, v in model_dict.items() if k not in ckp_dict]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('<Visual> load checkpoint from:', ckp_path)
    print('<Visual> unused param:', unused_param)
    print('<Visual> lost_param:', lost_param)

    if nlp_model is not None:
        model_dict = nlp_model.state_dict()
        if ckp.get('nlp_model', None) is None:
            raise Exception('use nlp model, but cannot find nlp_ckp')
        else:
            nlp_dict = ckp['nlp_model']

        pretrained_dict = {k: v for k, v in nlp_dict.items() if k in model_dict}
        unused_param = [k for k, v in nlp_dict.items() if k not in model_dict]
        lost_param = [k for k, v in model_dict.items() if k not in nlp_dict]

        nlp_dict.update(pretrained_dict)
        nlp_model.load_state_dict(model_dict)

        print('<NLP> load checkpoint from:', ckp_path)
        print('<NLP> unused param:', unused_param)
        print('<NLP> lost_param:', lost_param)
    else:
        pass


class Tester(object):
    def __init__(self, **kwargs):
        self.args = kwargs
        self.exp_cfg = kwargs.get('args', None)
        self.tester_cfg = kwargs.get('tester', None)
        self.tracker_cfg = kwargs.get('tracker', None)

        self.model = None
        self.nlp_model = None
        self.tracker = None

    def create_tracker(self):
        model_builder = self.args.get('model_builder', None)
        self.model = model_builder(self.exp_cfg.model).eval()

        if self.exp_cfg.model.use_language:
            nlp_module = importlib.import_module('MAT.lib.model.nlp_models')
            if self.exp_cfg.model.nlp_model.type == 'CLIP':
                self.nlp_model = getattr(nlp_module, self.exp_cfg.model.nlp_model.type)(
                    lr_mult=self.exp_cfg.model.nlp_model.lr_mult,
                    arch=self.exp_cfg.model.nlp_model.arch).eval()
            else:
                self.nlp_model = getattr(nlp_module, self.exp_cfg.model.nlp_model.type)(
                    lr_mult=self.exp_cfg.model.nlp_model.lr_mult).eval()

        if not os.path.exists(self.tracker_cfg.ckp_path):
            print('not find ckp path: {}'.format(self.tracker_cfg.ckp_path))
            raise AssertionError

        load_ckp(self.tracker_cfg.ckp_path, self.model, nlp_model=self.nlp_model)

        self.tracker = self.tracker_cfg.tracker_class(hyper=self.tracker_cfg, model=self.model)
        # tmp_tracker = MultiModalTracker(hyper=self.tracker_cfg, model=self.model)

    def model_to_device(self):
        if not next(self.model.parameters()).is_cuda:
            self.model.cuda()
            if self.exp_cfg.model.use_language:
                self.nlp_model.cuda()

            torch.cuda.empty_cache()

    def save_common(self, video_name, box_list, time_list, score_list):
        res_path = os.path.join(self.tester_cfg.res_dir, '{}.txt'.format(video_name))
        time_path = os.path.join(self.tester_cfg.res_dir, 'times', '{}_time.txt'.format(video_name))
        score_path = os.path.join(self.tester_cfg.res_dir, 'scores', '{}_confidence.txt'.format(video_name))

        np.savetxt(res_path, box_list, fmt='%.3f', delimiter=',')
        np.savetxt(time_path, time_list, fmt='%.8f', delimiter=',')
        np.savetxt(score_path, score_list, fmt='%.3f', delimiter=',')

    def save_got10k(self, video_name, box_list, time_list, score_list):
        os.makedirs(os.path.join(self.tester_cfg.res_dir, video_name), exist_ok=True)

        res_path = os.path.join(self.tester_cfg.res_dir, video_name, '{}_001.txt'.format(video_name))
        time_path = os.path.join(self.tester_cfg.res_dir, video_name, '{}_time.txt'.format(video_name))
        time2_path = os.path.join(self.tester_cfg.res_dir, 'times', '{}_time.txt'.format(video_name))
        score_path = os.path.join(self.tester_cfg.res_dir, 'scores', '{}_confidence.txt'.format(video_name))

        np.savetxt(res_path, box_list, fmt='%.3f', delimiter=',')
        np.savetxt(time_path, time_list, fmt='%.8f', delimiter=',')
        np.savetxt(time2_path, time_list, fmt='%.8f', delimiter=',')
        np.savetxt(score_path, score_list, fmt='%.3f', delimiter=',')

    def save_votlt(self, video_name, box_list, time_list, score_list):
        os.makedirs(os.path.join(self.tester_cfg.res_dir, 'longterm', video_name), exist_ok=True)

        res_path = os.path.join(self.tester_cfg.res_dir, 'longterm', video_name, '{}_001.txt'.format(video_name))
        score_path = os.path.join(self.tester_cfg.res_dir, 'longterm', video_name,
                                  '{}_001_confidence.value'.format(video_name))
        time_path = os.path.join(self.tester_cfg.res_dir, 'longterm', video_name, '{}_time.txt'.format(video_name))

        np.savetxt(res_path, box_list, fmt='%.3f', delimiter=',')
        with open(res_path, 'r') as f:
            ori = f.readlines()
            ori[0] = '1\n'
        with open(res_path, "w") as f:
            f.writelines(ori)

        np.savetxt(score_path, score_list, fmt='%.3f', delimiter=',')
        with open(score_path, 'r') as f:
            ori = f.readlines()
            ori[0] = '\n'
        with open(score_path, "w") as f:
            f.writelines(ori)

        np.savetxt(time_path, time_list, fmt='%.8f', delimiter=',')

    def run_ope(self):  # [x y x y]
        assert self.tester_cfg.num_gpu > 0, "need gpu for running"

        seqs = benchmark_loader[self.tester_cfg.benchmark]()
        self.create_tracker()

        if self.tester_cfg.num_process == 0:
            for video_idx, (video_name, im_list, gt_list, lang) in enumerate(seqs):
                seq_info = [video_idx, video_name, im_list, gt_list, lang, len(seqs)]
                self.run_seq(seq_info, self.tester_cfg.num_gpu)
        else:
            multiprocessing.set_start_method('spawn', force=True)
            print('>>> multi-processes running <<<')

            param_list = [([video_idx, video_name, im_list, gt_list, lang, len(seqs)],
                           self.tester_cfg.num_gpu)
                          for video_idx, (video_name, im_list, gt_list, lang) in enumerate(seqs)]

            with multiprocessing.Pool(processes=self.tester_cfg.num_process) as pool:
                pool.starmap(self.run_seq, param_list)

        eval_fun = benchmark_evaluator.get(self.tester_cfg.benchmark, None)
        if eval_fun is not None:
            eval_fun(self.tester_cfg.res_dir, self.tracker_cfg.name)

    def run_seq(self, seq_info, num_gpu):
        video_idx, video_name, im_list, gt_list, lang, seq_num = seq_info

        try:
            worker_name = multiprocessing.current_process().name
            worker_id = int(worker_name.split('-')[1]) - 1
            gpu_id = worker_id % num_gpu
            torch.cuda.set_device(gpu_id)
            rank = worker_id
            print('start rank {} on gpu {}'.format(rank, gpu_id))
        except IndexError:
            rank = 0
            print('Not multi-processes !')
            torch.cuda.set_device(0)

        # skip some videos
        if ('lasot' in self.tester_cfg.benchmark or 'tnl2k' in self.tester_cfg.benchmark
                or 'otb' in self.tester_cfg.benchmark or 'nfs' in self.tester_cfg.benchmark
                or 'uav123' in self.tester_cfg.benchmark
                or 'trackingnet' in self.tester_cfg.benchmark or 'otb99lang' in self.tester_cfg.benchmark
                or 'trek150' in self.tester_cfg.benchmark or 'itb' in self.tester_cfg.benchmark):
            if os.path.exists(os.path.join(self.tester_cfg.res_dir, '{}.txt'.format(video_name))):
                print('skip: ', self.tracker_cfg.name, '--', video_idx, video_name)
                return

        elif 'got10k' in self.tester_cfg.benchmark:
            if os.path.exists(os.path.join(self.tester_cfg.res_dir, video_name, '{}_001.txt'.format(video_name))):
                print('skip: ', self.tracker_cfg.name, '--', video_idx, video_name)
                return

        elif 'lt' in self.tester_cfg.benchmark:
            if os.path.exists(os.path.join(self.tester_cfg.res_dir, 'longterm', video_name,
                                           '{}_001.txt'.format(video_name))):
                print('skip: ', self.tracker_cfg.name, '--', video_idx, video_name)
                return
        else:
            raise NotImplementedError

        self.model_to_device()

        out_results = self.tracking_loop(im_list, gt_list, lang, video_name)

        box_list, score_list, time_list, fps_list = out_results

        if ('lasot' in self.tester_cfg.benchmark or 'tnl2k' in self.tester_cfg.benchmark
                or 'otb' in self.tester_cfg.benchmark or 'nfs' in self.tester_cfg.benchmark
                or 'uav123' in self.tester_cfg.benchmark
                or 'trackingnet' in self.tester_cfg.benchmark or 'otb99lang' in self.tester_cfg.benchmark
                or 'trek150' in self.tester_cfg.benchmark or 'itb' in self.tester_cfg.benchmark):
            self.save_common(video_name, box_list, time_list, score_list)

        elif 'got10k' in self.tester_cfg.benchmark:
            self.save_got10k(video_name, box_list, time_list, score_list)

        elif 'lt' in self.tester_cfg.benchmark:
            self.save_votlt(video_name, box_list, time_list, score_list)

        else:
            raise NotImplementedError

        print('[Rank {:2d}] {:3d}/{:d} {:<30s} -- [{:6.2f} fps]'.format(
            rank, video_idx, seq_num, video_name, np.mean(fps_list)))

    def tracking_loop(self, im_list, gt_list, lang, video_name):
        fps_list = np.zeros([len(im_list), 1])
        box_list = np.zeros([len(im_list), 4])
        score_list = np.zeros([len(im_list), 1])
        time_list = np.zeros([len(im_list), 1])

        if self.tracker_cfg.longterm:
            re_boxes = np.loadtxt(
                '/home/space/Documents/Official/mdetr/results/refcocog_EB3_checkpoint/{}.txt'.format(video_name),
                delimiter=',')

            # re_boxes = np.concatenate((
            #     np.array([gt_list[0]]),
            #     np.loadtxt('/home/space/Documents/Official/GlobalTrack/results/VOT19LT/longterm/{}/{}_001.txt'.format(
            #         video_name, video_name), delimiter=',', skiprows=1)
            # ), axis=0)
        else:
            re_boxes = None

        description_count = 0
        for im_i, im_f in enumerate(im_list):
            img = cv2.imread(im_f)
            gt = np.array(gt_list[0])  # [x y w h]

            tic = time.time()
            if im_i == 0:
                # word embedding
                if self.exp_cfg.model.use_language:
                    if isinstance(lang, list):
                        tmp = lang[im_i]
                    else:
                        tmp = lang
                    description_count += 1
                    lang_encode = self.nlp_model(tmp)  # (1, L, 768)
                else:
                    lang_encode = None

                self.tracker.init(img, gt, language=lang_encode)  # [x y w h]
                predict_box, predict_score = np.array(gt), 1

                delta_time = time.time() - tic
                fps_list[im_i, :] = 1 / delta_time
                time_list[im_i, :] = delta_time
                score_list[im_i, :] = predict_score
                box_list[im_i, :] = predict_box

            else:
                # word embedding
                if self.exp_cfg.model.use_language:
                    if isinstance(lang, list):
                        tmp = lang[im_i]
                        if len(tmp) > 0:
                            description_count += 1
                            # if (description_count - 1) % 4 == 0:
                            # if (description_count - 1) % 2 == 0:
                            # if (description_count - 0) % 4 != 0:
                            # if im_i % 2 == 0:
                            self.tracker.update_language(self.nlp_model(tmp))  # (1, L, 768)

                predict_box, predict_score, visualization = \
                    self.tracker.track(img, visualize=self.tracker_cfg.visualize)  # [x y w h]

                if self.tracker_cfg.longterm:
                    iou = box_iou(box_convert(torch.tensor(predict_box).unsqueeze(0), in_fmt='xywh', out_fmt='xyxy'),
                                  box_convert(torch.tensor(gt_list[im_i]).unsqueeze(0), in_fmt='xywh', out_fmt='xyxy'))
                    if max(iou.item(), 0) == 0:
                    # if predict_score < self.tracker_cfg.hyper:
                        self.tracker.re_detect(re_boxes[im_i])

                delta_time = time.time() - tic
                fps_list[im_i, :] = 1 / delta_time
                time_list[im_i, :] = delta_time
                score_list[im_i, :] = predict_score
                box_list[im_i, :] = predict_box

                if self.tracker_cfg.visualize:
                    att1, att2, att3, att4, attl, b_map, c_map = visualization
                    vis_dir = os.path.join(paths.tmp_dir, 'visualization_npy', video_name)
                    os.makedirs(vis_dir, exist_ok=True)
                    np.save(os.path.join(vis_dir, '{:0>6d}_attVb.npy'.format(im_i)), att1)
                    np.save(os.path.join(vis_dir, '{:0>6d}_attVc.npy'.format(im_i)), att2)
                    np.save(os.path.join(vis_dir, '{:0>6d}_attCb.npy'.format(im_i)), att3)
                    np.save(os.path.join(vis_dir, '{:0>6d}_attCc.npy'.format(im_i)), att4)
                    np.save(os.path.join(vis_dir, '{:0>6d}_attLang.npy'.format(im_i)), attl)
                    np.save(os.path.join(vis_dir, '{:0>6d}_predB.npy'.format(im_i)), b_map)
                    np.save(os.path.join(vis_dir, '{:0>6d}_predC.npy'.format(im_i)), c_map)

        return box_list, score_list, time_list, fps_list

#
# class GOT10kRunner(GOT10k_Tracker):
#     def __init__(self, args: dict):
#         super(GOT10kRunner, self).__init__(
#             name=args['tracker_name'],  # tracker name
#             is_deterministic=True  # stochastic (False) or deterministic (True)
#         )
#
#         model = MODEL_ARCH[args['model_arch']](MODEL_CFG[args['model_arch']])
#
#         if not os.path.exists(self.args.get('ckp_path', '')):
#             print('not find: {}'.format(self.args['ckp_path']))
#             raise AssertionError
#         load_ckp(model=model, ckp_path=self.args['ckp_path'])
#
#         self.tracker = BaseTracker(model=model, hyper=args)
#
#     def init(self, image, box):  # [x y w h]
#         im = np.array(image)  # to BGR opencv style
#         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#
#         gt = np.array(box)
#
#         self.tracker.init(im, gt)
#
#     def update(self, image):
#         im = np.array(image)  # to BGR opencv style
#         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#
#         predict_box, predict_score = self.tracker.track(im)  # [x y w h]
#
#         return predict_box


# def vot_run(args: dict):
#     from trackers import vot
#
#     model = MODEL_ARCH[args['model_arch']](MODEL_CFG[args['model_arch']])
#
#     if not os.path.exists(args.get('ckp_path', '')):
#         print('not find: {}'.format(args['ckp_path']))
#         raise AssertionError
#     load_ckp(model=model, ckp_path=args['ckp_path'])
#
#     tracker = BaseTracker(model=model, hyper=args)
#
#     handle = vot.VOT("rectangle")
#     region = handle.region()
#
#     image_file = handle.frame()
#     if not image_file:
#         sys.exit(0)
#
#     image = cv2.imread(image_file)
#     gt = np.array([region.x, region.y, region.width, region.height])
#     tracker.init(image, gt)  # [x y w h]
#
#     while True:
#         image_file = handle.frame()
#         if not image_file:
#             break
#         image = cv2.imread(image_file)
#
#         region, confidence = tracker.track(image)  # [x y w h]
#         region = vot.Rectangle(region[0], region[1], region[2], region[3])  # [x y w h]
#         handle.report(region, confidence)
#
#
# def create_workspace(
#         workspace,
#         project_path,
#         tag,
#         args,
#         stack,
# ):
#     os.makedirs(workspace, exist_ok=True)
#
#     with open(os.path.join(workspace, 'config.yaml'), 'w') as f:
#         f.write('registry:\n')
#         f.write('- ./trackers.ini\n')
#         f.write('stack: {}\n'.format(stack))
#
#     with open(os.path.join(workspace, 'trackers.ini'), 'w') as f:
#         f.write('[{}]\n'.format(tag))
#         f.write('label = {}\n'.format(tag))
#         f.write('protocol = traxpython\n')
#         if args is None:
#             f.write('command = from trackers.runner import VOT_Run; VOT_Run()\n')
#         else:
#             f.write('command = from trackers.runner import VOT_Run; VOT_Run({})\n'.format(
#                 f"args={{ \'model_arch\': \'{args['model_arch']}\', \'final_epoch\': {args['final_epoch']:d} }}")
#             )
#         f.write('paths = {}\n'.format(os.path.join(project_path)))
#         f.write('env_PATH = %s;%s;${PATH}\n' % (project_path, os.path.join(project_path, 'trackers')))
#
#     if not os.path.exists(os.path.join(workspace, 'sequences')):
#         os.system('ln -s {} {}'.format(paths.eval_vot20, os.path.join(workspace, 'sequences')))
