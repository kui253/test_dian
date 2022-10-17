_base_ = ['./resnet_strikes_back/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco.py']
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=10)))
classes = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9')
data = dict(

    train=dict(
        ann_file='D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/nums/train_coco.json',
        classes = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9'),
        img_prefix='D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/nums/mchar_train/'),
    val=dict(
        ann_file='D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/nums/val_coco.json',
        classes = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9'),
        img_prefix='D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/nums/mchar_val/'),
    test=dict(
        ann_file='D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/nums/val_coco.json',
        classes = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9'),
        img_prefix='D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/nums/mchar_val/'))
lr_config = None
load_from = "D:/anaconda3/envs/ve_first/Lib/site-packages/mmdet/.mim/checkpoints/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_162229-32ae82a9.pth"
