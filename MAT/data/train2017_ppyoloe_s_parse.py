import numpy as np
import json
from tqdm import tqdm

with open('./train2017_ppyoloe_s.json') as fin:
    tmp = json.load(fin)

id2path = tmp['id2path']

yoloe_dict = dict()
for d in tqdm(tmp['det']):
    id = int(d['im_id'][0][0])
    name = id2path[f'{id}'].split('/')[-1][:-4]
    bbox = np.array(d['bbox'])[:, 1:]
    if (bbox[:, 0] > 0.5).sum() > 0:
        bbox = bbox[bbox[:, 0] > 0.5].reshape(-1, 5)
    else:
        bbox = bbox

    bbox = bbox[:, 1:]
    bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
    yoloe_dict.update({name: bbox})


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode()
        else:
            return super(NpEncoder, self).default(obj)


json.dump(yoloe_dict, open('coco_train2017_ppyoloe_s.json', 'w'), indent=4, sort_keys=True, cls=NpEncoder)
