import torch
import os

def tracker_path_config(tracker_name):
    '''Configure Tracker:
    Configure this separately for all trackers'''

    base_folder = "trained_trackers"
    if tracker_name == "SiamRPN":
        model_config_path = f'{base_folder}/siamrpn_r50_l234_dwxcorr/config.yaml'
        model_path = f'{base_folder}/siamrpn_r50_l234_dwxcorr/model.pth'
    elif tracker_name == "SiamMASK":
        model_config_path = f'{base_folder}/siammask_r50_l3/config.yaml'
        model_path = f'{base_folder}/siammask_r50_l3/model.pth'
    elif tracker_name == "SiamBAN":
        model_config_path = f'{base_folder}/siamban_r50_l234/config.yaml'
        model_path = f'{base_folder}/siamban_r50_l234/model.pth'
    elif tracker_name == "SiamCAR":
        model_config_path = f'{base_folder}/siamcar_r50/config.yaml'
        model_path = f'{base_folder}/siamcar_r50/model_general.pth'
    elif tracker_name == "SiamFC":
        model_config_path = None
        model_path = 'trained_trackers/siamfc/model.pth'
    elif tracker_name == "DaSiamRPN":
        model_config_path = None
        model_path = f'{base_folder}/dasiamrpn/SiamRPNVOT.model'
    elif tracker_name == "ATOM":    
        model_config_path = 'default'
        model_path = f'{base_folder}/pytracking/atom_default.pth'
    elif tracker_name == "DiMP":    
        model_config_path = 'dimp18'
        model_path = f'{base_folder}/pytracking/dimp18.pth'
    elif tracker_name == "PrDiMP":    
        model_config_path = 'prdimp18'
        model_path = f'{base_folder}/pytracking/prdimp18.pth.tar'
    elif tracker_name == "SuperDiMP":    
        model_config_path = 'super_dimp'
        model_path = f'{base_folder}/pytracking/super_dimp.pth.tar'
    elif tracker_name == "KYS":    
        model_config_path = 'default'
        model_path = f'{base_folder}/pytracking/kys.pth'
    elif tracker_name == "KeepTrack":    
        model_config_path = 'default'
        model_path = f'{base_folder}/pytracking/keep_track.pth.tar'
    elif tracker_name == "ToMP":    
        model_config_path = 'tomp50'
        model_path = f'{base_folder}/pytracking/tomp.pth.tar'
    elif tracker_name == "RTS":    
        model_config_path = 'rts50'
        model_path = f'{base_folder}/pytracking/rts50.pth'
    elif tracker_name == "LWL":    
        model_config_path = 'lwl_boxinit'
        model_path = f'{base_folder}/pytracking/lwl_boxinit.pth'
    elif tracker_name == "STARK":
        model_config_path = 'baseline_got10k_only'
        model_path = f'{base_folder}/stark/got10k_only/STARKST_ep101.pth.tar'
    elif tracker_name == "TransT":
        folder = f'{base_folder}/TransT'
        model_config_path = 'TransT'
        model_path = f'{folder}/transt.pth'
    elif tracker_name == "TrDiMP":    
        model_config_path = 'trdimp'
        model_path = f'{base_folder}/TransformerTrack/trdimp_net.pth.tar'
    elif tracker_name == "TrSiam":   
        model_config_path = 'trsiam'
        model_path = f'{base_folder}/TransformerTrack/trdimp_net.pth.tar'
    elif tracker_name == "TrTr":
        model_config_path = None
        model_path = f'{base_folder}/TrTr/trtr_resnet50.pth'
    elif tracker_name == "SiamFCpp":
        model_config_path = f'{base_folder}/siamfcpp/siamfcpp_alexnet.yaml'
        model_path = f'{base_folder}/siamfcpp/siamfcpp-alexnet-vot-md5.pkl'
    elif tracker_name == "SparseTT":
        model_config_path = 'sparsett/experiments/sparsett/test/got10k/sparsett_swin_got10k.yaml'
        model_path = f'{base_folder}/sparsett/model_got10k.pkl'
    elif tracker_name == "SiamGAT":
        model_config_path = 'siamgat/experiments/siamgat_googlenet/config.yaml'
        model_path = f'{base_folder}/siamgat/otb_uav_model.pth'
    elif tracker_name == "SiamAttn":
        model_config_path = 'siamattn/experiments/config_vot2018.yaml'
        model_path = f'{base_folder}/siamattn/checkpoint_vot2018.pth'
    elif tracker_name == "CSWinTT":
        model_config_path = 'baseline_cs'
        model_path = f'{base_folder}/cswintt/CSWinTT.pth'
    elif tracker_name == "SiamRPN++-RBO":
        model_config_path = f'siamrpnpp_rbo/experiments/test/VOT2016/config.yaml'
        model_path = f'{base_folder}/siamrpnpp_rbo/SiamRPN++-RBO-general-OTNV.pth'
    elif tracker_name == "ARDiMP":
        model_config_path = None
        model_path = [f'{base_folder}/ardimp/super_dimp.pth.tar', 
                      f'{base_folder}/ardimp/SEcmnet_ep0040-c.pth.tar']
    elif tracker_name == "STMTrack":
        model_config_path = 'stmtrack/experiments/stmtrack/test/got10k/stmtrack-googlenet-got.yaml'
        model_path = f'{base_folder}/stmtrack/epoch-19_got10k.pkl'
    elif tracker_name == "STNet":
        model_config_path = 'stnet/experiments/test/fe240/fe240.yaml'
        model_path = f'{base_folder}/stnet/fe240.pkl'
    elif tracker_name == "AutoMatch":
        model_config_path = 'automatch/experiments/AutoMatch.yaml'
        model_path = f'{base_folder}/automatch/AutoMatch.pth'
    else:
        raise ValueError('No Matching Tracker Name')

    return model_config_path, model_path


def build_tracker(model_config_path, model_path, tracker_name):
    if tracker_name == "SiamFC":                                # SiamFC Tracker
        from siamfc.siamfc import TrackerSiamFC
        return TrackerSiamFC(model_path), 0
    elif tracker_name == "SiamBAN":                             # SiamBAN Tracker
        # load tracker config
        from siamban.core.config import cfg
        from siamban.models.model_builder import ModelBuilder
        from siamban.tracker.tracker_builder import build_tracker
        from siamban.utils.model_load import remove_prefix, check_keys

        cfg.merge_from_file(model_config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create  and load model
        model = ModelBuilder()

        pretrained_dict = torch.load(model_path,
            map_location=lambda storage, loc: storage.cpu())
        
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                            'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')

        try:
            check_keys(model, pretrained_dict)
        except:
            raise "Unable to check keys"
        model.load_state_dict(pretrained_dict, strict=False)

        model.eval().to(device)
        return build_tracker(model), 0 
    elif tracker_name == "SiamCAR":                         # SiamCar Tracker
        from siamcar.pysot.core.config import cfg 
        from siamcar.pysot.models.model_builder import ModelBuilder
        from siamcar.pysot.tracker.siamcar_tracker import SiamCARTracker
        from siamcar.utils.model_load import remove_prefix, check_keys

        cfg.merge_from_file(model_config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # hp_search
        params = getattr(cfg.HP_SEARCH, "OTB100")
        hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

        model = ModelBuilder()
        pretrained_dict = torch.load(model_path,
            map_location=lambda storage, loc: storage.cpu())
        
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                            'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')

        try:
            check_keys(model, pretrained_dict)
        except:
            raise "Unable to check keys"
        model.load_state_dict(pretrained_dict, strict=False)
        model.eval().to(device)

        cfg.TRACK.hanming = False
        return SiamCARTracker(model, cfg.TRACK), hp
    elif tracker_name == "DaSiamRPN":                       # DaSiamRPN Tracker
        from dasiamrpn.net import SiamRPNvot
        
        model = SiamRPNvot()
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, 
                loc: storage.cpu()))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval().to(device)
        return model, 0
    elif tracker_name == "SiamRPN" or tracker_name == "SiamMASK":   # SiamRPN and SiamMask Trackers
        import sys
        
        pp = f'{os.getcwd()}/pysot/'
        if not pp in sys.path:
            sys.path.insert(0, pp)
        if 'cfg' in locals():
            del cfg
            
        from pysot.core.config import cfg 
        from pysot.models.model_builder import ModelBuilder
        from pysot.tracker.tracker_builder import build_tracker

        cfg.merge_from_file(model_config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        model = ModelBuilder()
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, 
                loc: storage.cpu()))
        model.eval().to(device)
        return build_tracker(model), 0 
    elif tracker_name == "ATOM" or tracker_name == "KYS" or tracker_name == "KeepTrack" or \
         tracker_name == "DiMP" or tracker_name == "PrDiMP" or tracker_name == "SuperDiMP" or \
         tracker_name == "ToMP" or tracker_name == "RTS" or tracker_name == "LWL":     # Pytracking trackers

        from pytracking.evaluation import Tracker

        t_name = "keep_track" if tracker_name == "KeepTrack" else tracker_name.lower()

        if t_name == "dimp" or t_name == "prdimp" or t_name == "superdimp":
            tracker = Tracker("dimp", model_config_path, display_name=tracker_name) 
        else:
            tracker = Tracker(t_name, model_config_path, display_name=tracker_name)
        return customize_pytracking_tracker(tracker), 0     
    elif tracker_name == "STARK":     # Stark is also based on Pytracking
        import sys
        sys.path.insert(0, f'{os.getcwd()}/stark/')
        from stark.lib.test.evaluation.tracker import Tracker
        tracker = Tracker("stark_st", model_config_path, 'got', display_name=tracker_name)
        return customize_pytracking_tracker(tracker), 0
    elif tracker_name == "TransT":
        import sys
        sys.path.insert(0, f'{os.getcwd()}/transt/')
        
        from transt.pysot_toolkit.trackers.tracker import Tracker
        from transt.pysot_toolkit.trackers.net_wrappers import NetWithBackbone
        net = NetWithBackbone(net_path=model_path, use_gpu=True)
        tracker = Tracker(name='transt', net=net, window_penalty=0.49, \
            exemplar_size=128, instance_size=256)
        return tracker, 0
    elif tracker_name == "TrDiMP" or tracker_name == "TrSiam":
        import sys

        pp = 'pytracking'
        if pp in sys.path:
            sys.path.remove(pp)
            sys.path.remove('ltr')

        pp = f'{os.getcwd()}/TransformerTrack/'
        if not pp in sys.path:
            sys.path.insert(0, pp)

        from TransformerTrack.pytracking.evaluation import Tracker
        tracker = Tracker("trdimp", model_config_path, display_name=tracker_name)
        return customize_pytracking_tracker(tracker), 0
    elif tracker_name == "TrTr":
        import sys
        f'{os.getcwd()}/TrTr/'

        from TrTr.models.tracker import build_tracker as build_baseline_tracker
        from TrTr.models.hybrid_tracker import build_tracker as build_online_tracker
        
        parser = get_args_parser()
        args = parser.parse_args()

        args.tracker.checkpoint = model_path

        return build_baseline_tracker(args.tracker), 0
        #return build_online_tracker(args.tracker), 0
    elif tracker_name == "SiamFCpp":
        import sys
        f'{os.getcwd()}/siamfcpp/'

        from videoanalyst.config.config import cfg as root_cfg, specify_task
        from videoanalyst.model.builder import build as model_builder
        from videoanalyst.pipeline.builder import build as pipeline_builder
        
        root_cfg.merge_from_file(model_config_path)

        root_cfg.defrost()
        root_cfg.test.track.model.task_model.SiamTrack.pretrain_model_path = model_path

        #root_cfg = root_cfg.test
        task, task_cfg = specify_task(root_cfg.test)
        task_cfg.freeze()
        window_name = task_cfg.exp_name

        model = model_builder(task, task_cfg.model)
        pipeline = pipeline_builder(task, task_cfg.pipeline, model)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.set_device(dev)

        return pipeline, 0
    elif tracker_name == "SparseTT":
        import sys
        f'{os.getcwd()}/sparsett/'

        import os.path as osp
        from videoanalyst.config.config import cfg as root_cfg
        from videoanalyst.config.config import specify_task
        #from videoanalyst.engine.builder import build as tester_builder
        from videoanalyst.model import builder as model_builder
        from videoanalyst.pipeline import builder as pipeline_builder

        exp_cfg_path = osp.realpath(model_config_path)
        root_cfg.merge_from_file(exp_cfg_path)
        root_cfg = root_cfg.test

        root_cfg.defrost()
        root_cfg.track.model.task_model.SiamTrack.pretrain_model_path = model_path
        root_cfg.track.model.backbone.SwinTransformer.pretrained = \
                'trained_trackers/sparsett/swin_tiny_patch4_window7_224.pth'

        task, task_cfg = specify_task(root_cfg)
        task_cfg.freeze()
        torch.multiprocessing.set_start_method('spawn', force=True)

        model = model_builder.build("track", task_cfg.model)
        pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.set_device(dev)
        return pipeline, 0
    elif tracker_name == "STMTrack":
        import sys
        f'{os.getcwd()}/stmtrack/'

        import os.path as osp
        from stmtrack.videoanalyst.config.config import cfg as root_cfg
        from stmtrack.videoanalyst.config.config import specify_task
        #from videoanalyst.engine.builder import build as tester_builder
        from stmtrack.videoanalyst.model import builder as model_builder
        from stmtrack.videoanalyst.pipeline import builder as pipeline_builder
        from stmtrack.videoanalyst.utils import complete_path_wt_root_in_cfg

        exp_cfg_path = osp.realpath(model_config_path)
        root_cfg.merge_from_file(exp_cfg_path)

        root_cfg = complete_path_wt_root_in_cfg(root_cfg, os.getcwd())
        root_cfg = root_cfg.test

        root_cfg.defrost()
        root_cfg.track.model.task_model.STMTrack.pretrain_model_path = model_path

        task, task_cfg = specify_task(root_cfg)
        task_cfg.freeze()
        torch.multiprocessing.set_start_method('spawn', force=True)

        model = model_builder.build("track", task_cfg.model)
        pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.set_device(dev)
        return pipeline, 0
    elif tracker_name == "STNet":
        import sys
        f'{os.getcwd()}/stnet/'

        import os.path as osp
        from stnet.videoanalyst.config.config import cfg as root_cfg
        from stnet.videoanalyst.config.config import specify_task
        from stnet.videoanalyst.model import builder as model_builder
        from stnet.videoanalyst.pipeline import builder as pipeline_builder

        exp_cfg_path = osp.realpath(model_config_path)
        root_cfg.merge_from_file(exp_cfg_path)
        root_cfg = root_cfg.test

        root_cfg.defrost()
        root_cfg.track.model.task_model.SiamTrack.pretrain_model_path = model_path

        task, task_cfg = specify_task(root_cfg)
        task_cfg.freeze()
        torch.multiprocessing.set_start_method('spawn', force=True)

        model = model_builder.build("track", task_cfg.model)
        pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.set_device(dev)
        return pipeline, 0
    elif tracker_name == "SiamGAT":
        import sys
        f'{os.getcwd()}/siamgat/'
        from siamgat.pysot.core.config import cfg
        from siamgat.pysot.utils.model_load import load_pretrain
        from siamgat.pysot.models.model_builder_gat import ModelBuilder
        from siamgat.pysot.tracker.siamgat_tracker import SiamGATTracker

        cfg.merge_from_file(model_config_path)

        model = ModelBuilder()
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        model = load_pretrain(model, model_path)
        model.eval().to(device)

        params = getattr(cfg.HP_SEARCH, 'OTB100')
        cfg.TRACK.LR = params[0]
        cfg.TRACK.PENALTY_K = params[1]
        cfg.TRACK.WINDOW_INFLUENCE = params[2]

        tracker = SiamGATTracker(model)

        return tracker, 0
    elif tracker_name == "SiamAttn":
        import sys
        f'{os.getcwd()}/siamattn/'

        from siamattn.pysot.core.config import cfg
        from siamattn.pysot.models.model_builder import ModelBuilder
        from siamattn.pysot.tracker.tracker_builder import build_tracker
        from siamattn.pysot.utils.model_load import load_pretrain

        cfg.merge_from_file(model_config_path)
        model = ModelBuilder()
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        model = load_pretrain(model, model_path)
        model.eval().to(device)

        tracker = build_tracker(model)
        return tracker, 0
    elif tracker_name == "CSWinTT":     # Stark is also based on Pytracking
        import sys
        sys.path.insert(0, f'{os.getcwd()}/cswintt/')
        from cswintt.lib.test.evaluation.tracker import Tracker
        tracker = Tracker("cswintt", model_config_path, 'lasot', display_name=tracker_name)
        params = tracker.get_parameters()
        tracker = tracker.create_tracker(params)
        return tracker, 0
    elif tracker_name == "SiamRPN++-RBO":
        import sys
        f'{os.getcwd()}/siamrpnpp_rbo/'

        from siamrpnpp_rbo.pysot.core.config import cfg
        from siamrpnpp_rbo.pysot.models.model_builder import ModelBuilder
        from siamrpnpp_rbo.pysot.tracker.tracker_builder import build_tracker
        from siamrpnpp_rbo.pysot.utils.bbox import get_axis_aligned_bbox
        from siamrpnpp_rbo.pysot.utils.model_load import load_pretrain

        cfg.merge_from_file(model_config_path)
        model = ModelBuilder()
        model = load_pretrain(model, model_path).cuda().eval()
        tracker = build_tracker(model)

        return tracker, 0
    elif tracker_name == "ARDiMP":
        import sys
        f'{os.getcwd()}/ardimp/'

        #from ardimp.demo import get_dimp, get_ar
        # Get DiMP tracker

        from ardimp.pytracking.parameter.dimp.super_dimp_demo import parameters
        from ardimp.pytracking.tracker.dimp.dimp import DiMP

        params = parameters(model_path[0])
        params.visualization = True
        params.debug = False
        params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
        dimp_tracker = DiMP(params)

        # Get Refine module
        from ardimp.pytracking.refine_modules.refine_module import RefineModule
        selector_path = 0
        sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
        RF_module = RefineModule(model_path[1], selector_path, search_factor=sr, input_sz=input_sz)

        tracker = [dimp_tracker, RF_module]
        return tracker, 0
    elif tracker_name == "AutoMatch":
        import sys
        f'{os.getcwd()}/automatch/'

        import automatch.lib.tracker.sot_tracker as tracker_builder
        import automatch.lib.utils.model_helper as loader
        import automatch.lib.utils.sot_builder as builder
        import automatch.lib.utils.read_file as reader
        from easydict import EasyDict as edict

        config = edict(reader.load_yaml(model_config_path))
        siam_tracker = tracker_builder.SiamTracker(config)
        siambuilder = builder.Siamese_builder(config)
        siam_net = siambuilder.build()

        siam_net = loader.load_pretrain(siam_net, model_path, addhead=True, print_unuse=False)
        siam_net.eval()
        siam_net = siam_net.cuda()
        tracker = [siam_net, siam_tracker]
        return tracker, 0
    else:
        raise 'No Matching Tracker Name'


def customize_pytracking_tracker(tracker, debug=None, visdom_info=None):
    from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
    
    params = tracker.get_parameters()

    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    params.debug = debug_

    params.tracker_name = tracker.name
    params.param_name = tracker.parameter_name

    if tracker.display_name != 'STARK':
        tracker._init_visdom(visdom_info, debug_)

    multiobj_mode = getattr(params, 'multiobj_mode', getattr(tracker.tracker_class, \
        'multiobj_mode', 'default'))

    if multiobj_mode == 'default':
        tracker = tracker.create_tracker(params)
        if hasattr(tracker, 'initialize_features'):
            tracker.initialize_features()

    elif multiobj_mode == 'parallel':
        tracker = MultiObjectWrapper(tracker.tracker_class, params, tracker.visdom, \
             fast_load=True)
    else:
        raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

    return tracker


def get_args_parser():
    from jsonargparse import ArgumentParser, ActionParser
    from TrTr.models.hybrid_tracker import get_args_parser as tracker_args_parser

    parser = ArgumentParser(prog='demo')

    parser.add_argument('--use_baseline_tracker', action='store_true',
                        help='whether use baseline(offline) tracker')
    parser.add_argument('--video_name', default='', type=str,
                        help='empty to use webcam, otherwise *.mp4, *.avi, *jpg, *JPEG, or *.png are allowed')
    parser.add_argument('--debug', action='store_true',
                        help='whether visualize the debug result')

    parser.add_argument('--tracker', action=ActionParser(parser=tracker_args_parser()))

    return parser
