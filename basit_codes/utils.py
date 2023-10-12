COLOR = ((1, 0, 0), (1, 0, 1), (1, 1, 0), (0, 162/255, 232/255), (0.5, 0.5, 0.5),
         (0.5, 0.5, 0), (0, 0, 1), (0, 1, 1), (136/255, 0, 21/255), (255/255, 127/255, 39/255),
         (0, 0, 0), (0, 0, 0.5), (0, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0),
         (0.5, 0, 0.5), (0, 127/255, 21/255), (127/255, 39/255, 0), (1, 0.5, 0), (0.25, 0.25, 0.5),
         (1, 0.25, 0.25), (1, 0.5, 0.5), (0.5, 1, 0), (0.5, 1, 0.5), (0.5, 1, 1), (0.5, 1, 1),
         (0.5, 0, 0.25), (0.25, 1, 0.5), (0.5, 0.25, 0))

LINE_STYLE = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-', 
              '-', '--', ':', '-', '--', ':', '-', '--', ':', '-',
              '-', '--', ':', '-', '--', ':', '-', '--', ':', '-']

MARKER_STYLE = ['o', 'v', '<', '*', 'D', 'x', '.', 'x', '<', '.',
                '8', 's', 'p', 'h', 'H', '+', 'D', 'd', '1', '2',
                'o', 'v', '<', '*', 'D', 'x', '.', 'x', '<', '.']

from natsort import natsorted
from glob import glob
import numpy as np

def get_trackers_fps(trackers, track_time_dir, dataset_name):
    trackers_fps = {}
    for tracker in trackers:
        tracker_time_dir = f"{track_time_dir}/{dataset_name}/{tracker}"
        
        # List all tracking time files
        track_times_files = natsorted(glob(f"{tracker_time_dir}/*.txt"))
        
        all_track_times = []
        for t in track_times_files:
            with open(t) as f:
                track_times = f.readlines()
            
            # Append all times
            for time in track_times: 
                all_track_times.append(float(time.split('\n')[0])) 
            
        avg_track_time = sum(all_track_times)/len(all_track_times)
        avg_fps = np.around(1/avg_track_time, decimals=2)
        
        trackers_fps[tracker] = avg_fps
    
    return trackers_fps