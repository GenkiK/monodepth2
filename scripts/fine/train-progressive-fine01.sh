ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""


CUDA_VISIBLE_DEVICES=4 python -OO $ROOT_DIR/train_with_road_progressive_height.py --model_name person_car_annot_height_progressive_fine01 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.1 --annot_height


