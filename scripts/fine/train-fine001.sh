ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
CKPT_TIMESTAMP="05-23-18:09"

echo "script is runnning"
echo ""

# # HACK: this script is for resuming training
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --resume --ckpt_timestamp $CKPT_TIMESTAMP

CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height


