import _init_paths
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name1 = 'utb'
dataset_name2 = 'utb_sim'
dataset_name3 = 'utb_unsim'
"""ostrack"""
# trackers.extend(
# trackerlist(name='mostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_v2', dataset_name=dataset_name,
#         run_ids=None, display_name='OSTrack256'))

#  ostrack
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N_kf25', dataset_name=dataset_name1,
#                 run_ids=None, display_name='ostrack'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N_kf25', dataset_name=dataset_name2,
#                 run_ids=None, display_name='ostrack'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N_kf25', dataset_name=dataset_name3,
#                 run_ids=None, display_name='ostrack'))

# OStrack-N
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N', dataset_name=dataset_name1,
#                 run_ids=None, display_name='ostrack'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N', dataset_name=dataset_name2,
#                 run_ids=None, display_name='ostrack'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N', dataset_name=dataset_name3,
#                 run_ids=None, display_name='ostrack'))

# OStrack-N
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N_kf20', dataset_name=dataset_name1,
#                 run_ids=None, display_name='ostrack'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N_kf20', dataset_name=dataset_name2,
#                 run_ids=None, display_name='ostrack'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_N_kf20', dataset_name=dataset_name3,
#                 run_ids=None, display_name='ostrack'))

# AR
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_AR', dataset_name=dataset_name1,
#                 run_ids=None, display_name='ostrackkf'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_AR', dataset_name=dataset_name2,
#                 run_ids=None, display_name='ostrackkf'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_AR', dataset_name=dataset_name3,
#                 run_ids=None, display_name='ostrackkf'))

# AR - kf
trackers.extend(
    trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_AR_kf20', dataset_name=dataset_name1,
                run_ids=None, display_name='ostrackkf'))
trackers.extend(
    trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_AR_kf20', dataset_name=dataset_name2,
                run_ids=None, display_name='ostrackkf'))
trackers.extend(
    trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_AR_kf20', dataset_name=dataset_name3,
                run_ids=None, display_name='ostrackkf'))

# 384
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name1,
#                 run_ids=None, display_name='ostrackkf'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name2,
#                 run_ids=None, display_name='ostrackkf'))
# trackers.extend(
#     trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name3,
#                 run_ids=None, display_name='ostrackkf'))

#  stark_s
# baseline
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name1, run_ids=None,
#                             display_name='stark_s'))
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name2, run_ids=None,
#                             display_name='stark_s'))
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name3, run_ids=None,
#                             display_name='stark_s'))

# baseline-center-head
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline_center', dataset_name=dataset_name1, run_ids=None,
#                             display_name='stark_s center head'))
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline_center', dataset_name=dataset_name2, run_ids=None,
#                             display_name='stark_s center head'))
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline_center', dataset_name=dataset_name3, run_ids=None,
#                             display_name='stark_s center head'))

# baseline-center-head kf
# trackers.extend(
#     trackerlist(name='stark_s', parameter_name='baseline_center_kf40', dataset_name=dataset_name1, run_ids=None,
#                 display_name='stark_s center head kf'))
# trackers.extend(
#     trackerlist(name='stark_s', parameter_name='baseline_center_kf40', dataset_name=dataset_name2, run_ids=None,
#                 display_name='stark_s center head kf'))
# trackers.extend(
#     trackerlist(name='stark_s', parameter_name='baseline_center_kf40', dataset_name=dataset_name3, run_ids=None,
#                 display_name='stark_s center head kf'))

#  stark_st
# baseline 50
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name1, run_ids=None,
#                             display_name='stark_st'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name2, run_ids=None,
#                             display_name='stark_st'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name3, run_ids=None,
#                             display_name='stark_st'))

# baseline-center-head
# trackers.extend(
#     trackerlist(name='stark_st', parameter_name='baseline_center', dataset_name=dataset_name1, run_ids=None,
#                 display_name='stark_st'))
# trackers.extend(
#     trackerlist(name='stark_st', parameter_name='baseline_center', dataset_name=dataset_name2, run_ids=None,
#                 display_name='stark_st'))
# trackers.extend(
#     trackerlist(name='stark_st', parameter_name='baseline_center', dataset_name=dataset_name3, run_ids=None,
#                 display_name='stark_st'))

# baseline-center-head-kf
# trackers.extend(
#     trackerlist(name='stark_st', parameter_name='baseline_center_kf40', dataset_name=dataset_name1, run_ids=None,
#                 display_name='stark_st'))
# trackers.extend(
#     trackerlist(name='stark_st', parameter_name='baseline_center_kf40', dataset_name=dataset_name2, run_ids=None,
#                 display_name='stark_st'))
# trackers.extend(
#     trackerlist(name='stark_st', parameter_name='baseline_center_kf40', dataset_name=dataset_name3, run_ids=None,
#                 display_name='stark_st'))

#  Tomp
# trackers.extend(trackerlist(name='tomp', parameter_name='tomp50', dataset_name=dataset_name1, run_ids=None,
#                             display_name='tomp'))
# trackers.extend(trackerlist(name='tomp', parameter_name='tomp50', dataset_name=dataset_name1, run_ids=None,
#                             display_name='tomp'))
# trackers.extend(trackerlist(name='tomp', parameter_name='tomp50', dataset_name=dataset_name3, run_ids=None,
#                             display_name='tomp'))
#  TrDimp
# trackers.extend(trackerlist(name='trdimp', parameter_name='trdimp', dataset_name=dataset_name1, run_ids=None,
#                             display_name='trdimp'))
# trackers.extend(trackerlist(name='trdimp', parameter_name='trdimp', dataset_name=dataset_name2, run_ids=None,
#                             display_name='trdimp'))
# trackers.extend(trackerlist(name='trdimp', parameter_name='trdimp', dataset_name=dataset_name3, run_ids=None,
#                             display_name='trdimp'))
#  Dimp
# trackers.extend(trackerlist(name='dimp', parameter_name='dimp50', dataset_name=dataset_name1, run_ids=None,
#                             display_name='dimp'))
# trackers.extend(trackerlist(name='dimp', parameter_name='dimp50', dataset_name=dataset_name2, run_ids=None,
#                             display_name='dimp'))
# trackers.extend(trackerlist(name='dimp', parameter_name='dimp50', dataset_name=dataset_name3, run_ids=None,
#                             display_name='dimp'))
#  keep-track
# trackers.extend(trackerlist(name='keep_track', parameter_name='default', dataset_name=dataset_name1, run_ids=None,
#                             display_name='keep_track'))
# trackers.extend(trackerlist(name='keep_track', parameter_name='default', dataset_name=dataset_name2, run_ids=None,
#                             display_name='keep_track'))
# trackers.extend(trackerlist(name='keep_track', parameter_name='default', dataset_name=dataset_name3, run_ids=None,
#                             display_name='keep_track'))
#  transt
# trackers.extend(trackerlist(name='transt', parameter_name='baseline', dataset_name=dataset_name1, run_ids=None,
#                             display_name='transt'))
# trackers.extend(trackerlist(name='transt', parameter_name='baseline', dataset_name=dataset_name2, run_ids=None,
#                             display_name='transt'))
# trackers.extend(trackerlist(name='transt', parameter_name='baseline', dataset_name=dataset_name3, run_ids=None,
#                             display_name='transt'))

# trackers.extend(trackerlist(name='transt', parameter_name='baseline_kf20', dataset_name=dataset_name1, run_ids=None,
#                             display_name='transt'))
# trackers.extend(trackerlist(name='transt', parameter_name='baseline_kf20', dataset_name=dataset_name2, run_ids=None,
#                             display_name='transt'))
# trackers.extend(trackerlist(name='transt', parameter_name='baseline_kf20', dataset_name=dataset_name3, run_ids=None,
#                             display_name='transt'))
#  siamcar
# trackers.extend(trackerlist(name='siamcar', parameter_name='baseline', dataset_name=dataset_name1,
#                             run_ids=None, display_name='baseline'))
# trackers.extend(trackerlist(name='siamcar', parameter_name='baseline', dataset_name=dataset_name2,
#                             run_ids=None, display_name='baseline'))
# trackers.extend(trackerlist(name='siamcar', parameter_name='baseline', dataset_name=dataset_name3,
#                             run_ids=None, display_name='baseline'))

#  aiatrack
# trackers.extend(trackerlist(name='aiatrack', parameter_name='baseline', dataset_name=dataset_name1,
#                             run_ids=None, display_name='aiatrack'))
# trackers.extend(trackerlist(name='aiatrack', parameter_name='baseline', dataset_name=dataset_name2,
#                             run_ids=None, display_name='aiatrack'))
# trackers.extend(trackerlist(name='aiatrack', parameter_name='baseline', dataset_name=dataset_name3,
#                             run_ids=None, display_name='aiatrack'))
#  mixformer_online
# trackers.extend(trackerlist(name='mixformer', parameter_name='baseline', dataset_name=dataset_name1,
#                             run_ids=None, display_name='mixformer_online'))
# trackers.extend(trackerlist(name='mixformer', parameter_name='baseline', dataset_name=dataset_name2,
#                             run_ids=None, display_name='mixformer_online'))
# trackers.extend(trackerlist(name='mixformer', parameter_name='baseline', dataset_name=dataset_name3,
#                             run_ids=None, display_name='mixformer_online'))
#  atom
# trackers.extend(trackerlist(name='atom', parameter_name='default', dataset_name=dataset_name1, run_ids=None,
#                             display_name='atom'))
# trackers.extend(trackerlist(name='atom', parameter_name='default', dataset_name=dataset_name2, run_ids=None,
#                             display_name='atom'))
# trackers.extend(trackerlist(name='atom', parameter_name='default', dataset_name=dataset_name3, run_ids=None,
#                              display_name='atom'))
#  siamban-acm
# trackers.extend(trackerlist(name='siamban_acm', parameter_name='baseline', dataset_name=dataset_name1, run_ids=None,
#                             display_name='siamban_acm'))
# trackers.extend(trackerlist(name='siamban_acm', parameter_name='baseline', dataset_name=dataset_name2, run_ids=None,
#                             display_name='siamban_acm'))
# trackers.extend(trackerlist(name='siamban_acm', parameter_name='baseline', dataset_name=dataset_name3, run_ids=None,
#                              display_name='siamban_acm'))
#  siamban
# trackers.extend(trackerlist(name='siamban', parameter_name='baseline', dataset_name=dataset_name1, run_ids=None,
#                             display_name='siamban'))
# trackers.extend(trackerlist(name='siamban', parameter_name='baseline', dataset_name=dataset_name2, run_ids=None,
#                             display_name='siamban'))
# trackers.extend(trackerlist(name='siamban', parameter_name='baseline', dataset_name=dataset_name3, run_ids=None,
#                              display_name='siamban'))

dataset1 = get_dataset(dataset_name1)
dataset2 = get_dataset(dataset_name2)
dataset3 = get_dataset(dataset_name3)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset1, dataset_name1, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
print_results(trackers, dataset2, dataset_name2, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
print_results(trackers, dataset3, dataset_name3, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
