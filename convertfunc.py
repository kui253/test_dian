import os.path as osp
import json
import mmcv
def convert2coco(ann_file,out_file,image_prefix):
    data_infos = mmcv.load(ann_file)#得到字典类型的对象
    annotations = []
    images = []
    ann_num = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = '%06d.png' % int(idx)
        img_path = osp.join(image_prefix,filename)
        height,width = mmcv.imread(img_path).shape[:2]#取出前两个元素
        images.append(dict(
            id = idx,
            file_name = filename,
            height = height,
            width = width
        ))
        sig_label = v['label']

        for i in range(len(sig_label)):
            sig_bbox = [v['left'][i],v['top'][i],v['width'][i],v['height'][i]]
            ann_dict = {
                'id':ann_num,
                'image_id':idx,
                'category_id':sig_label[i],
                'bbox':sig_bbox,
                'area':v['width'][i]*v['height'][i],
                'iscrowd':0
            }
            annotations.append(ann_dict)
            ann_num += 1
    
    coco_output = dict(
        images = images,
        annotations = annotations,
        categories = [
            {'id':0,'name':'0'},
            {'id':1,'name':'1'},
            {'id':2,'name':'2'},
            {'id':3,'name':'3'},
            {'id':4,'name':'4'},
            {'id':5,'name':'5'},
            {'id':6,'name':'6'},
            {'id':7,'name':'7'},
            {'id':8,'name':'8'},
            {'id':9,'name':'9'},
        ]
    )
    json.dump(coco_output,open(out_file,'w'))
ann_file = 'mchar_train.json'
out_file = 'train_coco.json'
img_pre = './mchar_train/'
ann_file2 = 'mchar_val.json'
out_file2 = 'val_coco.json'
img_pre2 = './mchar_val/'
convert2coco(ann_file2,out_file2,img_pre2)
