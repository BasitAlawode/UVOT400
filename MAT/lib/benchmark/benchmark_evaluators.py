import os

from got10k.experiments import ExperimentGOT10k, ExperimentLaSOT, ExperimentOTB
from register import path_register


def eval_lasot(save_dir, tracker_name):
    """

    Args:
        save_dir: ./results/<benchmark_name>/<tracker_name>
        tracker_name: <tracker_name>
    """
    experiment = ExperimentLaSOT(
        root_dir=os.path.join(path_register.benchmark.lasot),
        result_dir='./',
        report_dir='./',
        subset='test',  # 'train' | 'val' | 'test'
    )
    experiment.result_dir = save_dir.replace(tracker_name, '')
    experiment.report_dir = save_dir.replace('results', 'reports').replace(tracker_name, '')

    # os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
    # runner = GOT10kRunner(args)
    # experiment.run(runner, visualize=False)
    experiment.report([tracker_name])


def eval_otb(save_dir, tracker_name):
    """

    Args:
        save_dir: ./results/<benchmark_name>/<tracker_name>
        tracker_name: <tracker_name>
    """
    experiment = ExperimentOTB(
        root_dir=path_register.benchmark.otb,
        result_dir='./',
        report_dir='./',
    )
    experiment.result_dir = save_dir.replace(tracker_name, '')
    experiment.report_dir = save_dir.replace('results', 'reports').replace(tracker_name, '')

    # os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
    # runner = GOT10kRunner(args)
    # experiment.run(runner, visualize=False, overwrite_result=False)
    experiment.report([tracker_name])


def eval_got10k_val(save_dir, tracker_name):
    """

    Args:
        save_dir: ./results/<benchmark_name>/<tracker_name>
        tracker_name: <tracker_name>
    """
    experiment = ExperimentGOT10k(
        root_dir=os.path.join(path_register.benchmark.got10k_val, '..'),
        result_dir='./',
        report_dir='./',
        subset='val',  # 'train' | 'val' | 'test'
    )
    experiment.result_dir = save_dir.replace(tracker_name, '')
    experiment.report_dir = save_dir.replace('results', 'reports').replace(tracker_name, '')

    # os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
    # runner = GOT10kRunner(args)
    # experiment.run(runner, visualize=False, overwrite_result=False)
    experiment.report([tracker_name])


def eval_got10k_test(save_dir, tracker_name):
    """

    Args:
        save_dir: ./results/<benchmark_name>/<tracker_name>
        tracker_name: <tracker_name>
    """
    # experiment = ExperimentGOT10k(
    #     root_dir=os.path.join(path_register.benchmark.got10k_test, '..'),
    #     result_dir='./',
    #     report_dir='./',
    #     subset='test',  # 'train' | 'val' | 'test'
    # )
    # experiment.result_dir = save_dir.replace(tracker_name, '')
    # experiment.report_dir = save_dir.replace('results', 'reports').replace(tracker_name, '')
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
    # runner = GOT10kRunner(args)
    # experiment.run(runner, visualize=False, overwrite_result=False)
    # experiment.report([tracker_name])

    result_path = save_dir.replace(tracker_name, '')
    report_path = save_dir.replace('results', 'reports').replace(tracker_name, '')
    report_file = save_dir.replace('results', 'reports')

    os.makedirs(report_path, exist_ok=True)
    os.chdir(result_path)
    os.system('zip -r {}.zip {}'.format(report_file, tracker_name))


def eval_trackingnet(save_dir, tracker_name):
    """

    Args:
        save_dir: ./results/<benchmark_name>/<tracker_name>
        tracker_name: <tracker_name>
    """
    # experiment = ExperimentTrackingNet(
    #     root_dir=os.path.join(path_register.benchmark.trackingnet),
    #     result_dir='./',
    #     report_dir='./',
    #     subset='test',
    # )
    # experiment.result_dir = save_dir.replace(tracker_name, '')
    # experiment.report_dir = save_dir.replace('results', 'reports').replace(tracker_name, '')
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
    # runner = GOT10kRunner(args)
    # experiment.run(runner, visualize=False)

    result_path = save_dir.replace(tracker_name, '')
    report_path = save_dir.replace('results', 'reports').replace(tracker_name, '')
    report_file = save_dir.replace('results', 'reports')

    os.makedirs(report_path, exist_ok=True)
    os.chdir(result_path)
    os.system('zip -r {}.zip {}'.format(report_file, tracker_name))
