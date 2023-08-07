# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamrpnpp_rbo.pysot.core.config import cfg
from siamrpnpp_rbo.pysot.tracker.siamrpn_tracker import SiamRPNTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
