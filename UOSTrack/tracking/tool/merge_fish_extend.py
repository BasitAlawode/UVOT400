import json

a = r'D:\BaiduNetdiskDownload\fish\Fishseq\fishclean\annotations\train.json'

c = r'D:\BaiduNetdiskDownload\fish\Fishseq\fishv1seq\annotations\train.json'

save = r'D:\BaiduNetdiskDownload\fish\Fishseq\fishextand\train.json'

with open(a, 'r') as f:
    train_json = json.load(f)

with open(c, 'r') as f:
    valid_json = json.load(f)

train_imgs_num = len(train_json['images'])
train_anno_num = len(train_json['annotations'])

valid_imgs_num = len(valid_json['images'])
valid_anno_num = len(valid_json['annotations'])


# merge valid
for num in range(valid_imgs_num):
    new_dict = valid_json['images'][num]
    new_dict['id'] += train_imgs_num
    train_json['images'].append(new_dict)

# anno
for num in range(valid_anno_num):
    new_dict = valid_json['annotations'][num]
    new_dict['id'] += train_anno_num
    new_dict['image_id'] += train_imgs_num
    new_dict['category_id'] = 0
    train_json['annotations'].append(new_dict)

with open(save, 'w') as f:
    json.dump(train_json, f)


