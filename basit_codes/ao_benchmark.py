import numpy as np

from colorama import Style, Fore

from toolkit.utils import overlap_ratio

class AOBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh+1e-16)

    def eval_ao(self, eval_trackers=None):
        """
        Args: 
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        ao_ret = {}
        for tracker_name in eval_trackers:
            ao_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                ao_ret_[video.name] = np.mean(overlap_ratio(gt_traj, tracker_traj))
            ao_ret[tracker_name] = ao_ret_
        return ao_ret

    def eval_sr50(self, eval_trackers=None):
        """
        Args: 
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        sr50_ret = {}
        for tracker_name in eval_trackers:
            sr50_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                sr50_ret_[video.name] = np.mean(overlap_ratio(gt_traj, tracker_traj) > 0.5)
            sr50_ret[tracker_name] = sr50_ret_
        return sr50_ret
    
    def eval_sr75(self, eval_trackers=None):
        """
        Args: 
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        sr75_ret = {}
        for tracker_name in eval_trackers:
            sr75_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                sr75_ret_[video.name] = np.mean(overlap_ratio(gt_traj, tracker_traj) > 0.75)
            sr75_ret[tracker_name] = sr75_ret_
        return sr75_ret

    def show_result(self, ao_ret, sr50_ret=None,
            sr75_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_ao = {}
        for tracker_name in ao_ret.keys():
            ao = np.mean(list(ao_ret[tracker_name].values()))
            tracker_ao[tracker_name] = ao
        #tracker_auc_ = sorted(tracker_auc.items(),
        #                     key=lambda x:x[1],
        #                     reverse=True)[:20]
        tracker_auc_ = sorted(tracker_ao.items(),
                             key=lambda x:x[1],
                             reverse=True)
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in ao_ret.keys()])+2), 12)
        header = ("|{:^"+str(tracker_name_len)+"}|{:^9}|{:^16}|{:^11}|").format(
                "Tracker name", "AO", "SR0.5", "SR0.75")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            ao = tracker_ao[tracker_name]
            if sr50_ret is not None:
                sr50 = np.mean(list(sr50_ret[tracker_name].values())) #[20]
            else:
                sr50 = 0
            if sr75_ret is not None:
                sr75 = np.mean(list(sr75_ret[tracker_name].values()),
                        axis=0)  #[20]
            else:
                sr75 = 0
            print(formatter.format(tracker_name, ao, sr50, sr75))
        print('-'*len(header))

        if show_video_level and len(ao_ret) < 10 \
                and sr50_ret is not None \
                and len(sr50_ret) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in ao_ret.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("ao", "sr0.5")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(ao_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in ao_ret.keys():
                    success = np.mean(ao_ret[tracker_name][video])
                    precision = np.mean(sr50_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < helight_threshold:
                        row += f'{Fore.RED}{success_str}{Style.RESET_ALL}|'
                    else:
                        row += success_str+'|'
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f'{Fore.RED}{precision_str}{Style.RESET_ALL}|'
                    else:
                        row += precision_str+'|'
                print(row)
            print('-'*len(header1))
