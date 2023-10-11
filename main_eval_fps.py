from glob import glob
from natsort import natsorted
import numpy as np
from main_eval import trackers

dataset_name = "UTB400"
track_time_dir = "trackers_times"

def get_trackers_fps(trackers):
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

if __name__ == "__main__":
    trackers_fps = get_trackers_fps(trackers)
    
    for tracker in trackers_fps.keys():
        print(f"{tracker}:   {trackers_fps[tracker]} fps")
    
    