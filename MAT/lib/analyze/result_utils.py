import os
import numpy as np
from tqdm import tqdm


def results_loader(benchmark, seq_list, result_dir, return_st=True):
    results_dict = dict()

    time_path = None
    score_path = None
    for (video_name, im_list, gt_list, lang) in tqdm(seq_list, ascii=True, desc=f'loading from {result_dir}'):

        if ('lasot' in benchmark or 'tnl2k' in benchmark
                or 'trackingnet' in benchmark or 'otb99lang' in benchmark
                or 'trek150' in benchmark):
            res_path = os.path.join(result_dir, '{}.txt'.format(video_name))
            time_path = os.path.join(result_dir, 'times', '{}_time.txt'.format(video_name))
            score_path = os.path.join(result_dir, 'scores', '{}_confidence.txt'.format(video_name))

        elif 'got10k' in benchmark:
            res_path = os.path.join(result_dir, video_name, '{}_001.txt'.format(video_name))
            time_path = os.path.join(result_dir, video_name, '{}_time.txt'.format(video_name))

            # time_path = os.path.join(result_dir, 'times', '{}_time.txt'.format(video_name))
            # score_path = os.path.join(result_dir, 'scores', '{}_confidence.txt'.format(video_name))

        elif 'lt' in benchmark:
            res_path = os.path.join(result_dir, 'longterm', video_name, '{}_001.txt'.format(video_name))
            score_path = os.path.join(result_dir, 'longterm', video_name,
                                      '{}_001_confidence.value'.format(video_name))
            time_path = os.path.join(result_dir, 'longterm', video_name, '{}_time.txt'.format(video_name))

        else:
            raise NotImplementedError

        gts = np.array(gt_list)

        if return_st:
            if 'lt' not in benchmark:
                try:
                    boxes = np.loadtxt(res_path, delimiter=',')
                except:
                    boxes = np.loadtxt(res_path, delimiter='\t')

                if score_path is not None:
                    scores = np.loadtxt(score_path, delimiter=',')
                else:
                    scores = None

                if time_path is not None:
                    times = np.loadtxt(time_path, delimiter=',')
                else:
                    times = None

            else:
                boxes = np.loadtxt(res_path, delimiter=',', skiprows=1)
                scores = np.loadtxt(score_path, delimiter=',', skiprows=1)
                times = np.loadtxt(time_path, delimiter=',', skiprows=1)

                boxes = np.concatenate((gts[:1], boxes), axis=0)
                scores = np.concatenate((np.ones([1, 1]), scores), axis=0)
                times = np.concatenate((times[:1], times), axis=0)

            results_dict.update({video_name: [boxes, scores, times]})

        else:
            if 'lt' not in benchmark:
                try:
                    boxes = np.loadtxt(res_path, delimiter=',')
                except:
                    boxes = np.loadtxt(res_path, delimiter='\t')

            else:
                boxes = np.loadtxt(res_path, delimiter=',', skiprows=1)

                boxes = np.concatenate((gts[:1], boxes), axis=0)

            results_dict.update({video_name: [boxes]})

    return results_dict
