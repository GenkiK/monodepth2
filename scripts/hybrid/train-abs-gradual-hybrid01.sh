ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
WEIGHT=0.1

echo "script is runnning"
echo ""

# CUDA_VISIBLE_DEVICES=3 python -OO $ROOT_DIR/train_hybrid.py --model_name "person_car_annot_height_abs_gradual_fine_rough$WEIGHT" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $WEIGHT --rough_metric_scale_weight $WEIGHT --annot_height --gradual_metric_scale_weight --cam_height_loss_func "abs"


# disable automasking
CUDA_VISIBLE_DEVICES=3 python -OO $ROOT_DIR/train_hybrid.py --model_name "person_car_annot_height_abs_disable_mask_gradual_fine_rough$WEIGHT" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $WEIGHT --rough_metric_scale_weight $WEIGHT --annot_height --gradual_metric_scale_weight --cam_height_loss_func "abs" --disable_automasking


# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_hybrid.py --model_name "person_car_annot_height_abs_gradual_fine_rough$WEIGHT" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $WEIGHT --rough_metric_scale_weight $WEIGHT --annot_height --gradual_metric_scale_weight --cam_height_loss_func "abs"
