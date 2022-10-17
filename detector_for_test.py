from mmdet.apis import init_detector, inference_detector
import numpy as np
import os
# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/configs/resnet_strikes_back/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco.py'
checkpoint_file = 'C:/Users/夔whd/work_dirs/nums/latest.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cpu')

# 测试单张图片并展示结果
img = 'D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/configs/test/'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# 在一个新的窗口中将结果可视化
#内部循环
lis_out = []
j = 0
for image_name in os.listdir(img):
    result = inference_detector(model, img+image_name)
    i = 0
    thr_list = [] 
    
    for class_result in result:
        if len(class_result)!= 0:
            
            class_list = class_result[0]
            if(class_list[4]>= 0.8):
                thr_list.append([class_list[0],Class_name[i]])
        i += 1
        if i>16:
            break
    if len(thr_list)==0:
        ls = ['none']
        lis_out.append(['%06d.png' % int(j),ls])
    else:
        ls = sorted(thr_list,key = lambda x:x[0])
        ls2 = ls[0][1]
        for i in range(np.shape(ls)[0]-1):
            ls2 += ls[i+1][1]
        lis_out.append(['%06d.png' % int(j),ls2])
    j+=1
print(lis_out)

