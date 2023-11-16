import os

import registers.path_register as paths
from trackers.runner import Runner, create_workspace


def run_ope(args: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
    runner = Runner(args)
    runner.run_ope(benchmark_name=args['benchmark_name'],
                   benchmark_root=args['benchmark_root'],
                   save_dir=args['save_dir'],
                   need_language=args.get('need_language', False),
                   parallel=args.get('max_process', 0)
                   )


# def run_lasot(args: dict):
#     os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
#
#     runner = GOT10kRunner(args)
#
#     # setup experiment (validation subset)
#     experiment = ExperimentLaSOT(
#         root_dir=os.path.join(paths.eval_lasot),
#         result_dir=os.path.join(paths.result_dir, 'LaSOT'),  # where to store tracking results
#         report_dir=os.path.join(paths.report_dir, 'LaSOT'),       # where to store evaluation reports
#         subset='test',  # 'train' | 'val' | 'test'
#     )
#     experiment.run(runner, visualize=False)
#     experiment.report([args['tracker_name']])


# def run_got10k(args: dict):
#     os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
#
#     runner = GOT10kRunner(args)
#
#     # setup experiment (validation subset)
#     experiment = ExperimentGOT10k(
#         root_dir=os.path.join(paths.eval_got10k_test, '..'),       # GOT-10k's root directory
#         result_dir=os.path.join(paths.result_dir,
#                                 'GOT10k_{}'.format(args['subset'])),  # where to store tracking results
#         report_dir=os.path.join(paths.report_dir, 'GOT10k'),       # where to store evaluation reports
#         subset=args['subset'],  # 'train' | 'val' | 'test'
#     )
#     experiment.run(runner, visualize=False, overwrite_result=False)
#     experiment.report([args['tracker_name']])


# def run_trackingnet(args: dict):
#     os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
#
#     runner = GOT10kRunner(args)
#
#     # setup experiment (validation subset)
#     experiment = ExperimentTrackingNet(
#         root_dir=os.path.join(paths.eval_trackingnet, '..'),       # GOT-10k's root directory
#         result_dir=os.path.join(paths.result_dir, 'TrackingNet'),  # where to store tracking results
#         report_dir=os.path.join(paths.report_dir, 'TrackingNet'),       # where to store evaluation reports
#         subset='test',
#     )
#     experiment.run(runner, visualize=False)
#     res_path = os.path.join(paths.result_dir, 'TrackingNet', args['tracker_name'])
#     os.system('zip -r {}.zip {}'.format(res_path, res_path))
#     # experiment.report([args['tracker_name']])  #


def run_vot(args: dict):
    workspace = os.path.join(paths.vot_workspace_dir, args['tracker_name'])

    create_workspace(workspace=workspace,
                     project_path=paths.project_dir,
                     tag=args['tracker_name'], stack=args['stack'], args=args)

    print('CUDA_VISIBLE_DEVICES={0} vot evaluate {1} --workspace {2}'.format(
        args['gpu_id'], args['tracker_name'], workspace))
    os.system('CUDA_VISIBLE_DEVICES={0} vot evaluate {1} --workspace {2}'.format(
        args['gpu_id'], args['tracker_name'], workspace))
    os.system('vot analysis {0} --workspace {1}'.format(args['tracker_name'], workspace))


RUN_ON_BENCHMARK = {
    'LaSOT': run_ope,
    'TrackingNet': run_ope,
    'TNL2K': run_ope,
    'VOT2020': run_vot,
    'GOT10k_test': run_ope,
    'GOT10k_val': run_ope,
    'VID_Sent': run_ope,
    'VOT2019-LT': run_ope,
    'VOT2018-LT': run_ope,
    'OTB_Lang': run_ope,
}
