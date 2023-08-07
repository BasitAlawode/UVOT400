from stark.lib.test.evaluation.environment import EnvSettings
import os

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = f'{os.getcwd()}/trained_trackers/stark/got10k_only/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = f'{os.getcwd()}/stark/lib/test/result_plots/'
    settings.results_path = f'{os.getcwd()}/stark/lib/test/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = f'{os.getcwd()}/stark/lib/test/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    settings.prj_dir = f'{os.getcwd()}/stark'
    settings.save_dir = settings.network_path

    return settings

