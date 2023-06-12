ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
WEIGHT=0.5

echo "script is runnning"
echo ""

CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name "person_car_annot_height_gradual_fine_rough$WEIGHT" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $WEIGHT --rough_metric_scale_weight $WEIGHT --annot_height --gradual_metric_scale_weight
