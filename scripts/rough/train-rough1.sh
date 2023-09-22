ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

ROUGH_WEIGHT=1

# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_segm.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough1 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10 --rough_metric_scale_weight 1.0

# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_with_segm.py --model_name person_car_annot_height_disable_mask_rough1 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car --rough_metric_scale_weight 1.0 --annot_height --disable_automasking

# abs
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_segm.py --model_name person_car_annot_height_abs_rough1 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car --rough_metric_scale_weight 1.0 --annot_height --rough_metric_loss_func "abs"

# mean_after_abs
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_segm.py --model_name person_car_annot_height_mean_after_abs_rough1 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car --rough_metric_scale_weight 1.0 --annot_height --rough_metric_loss_func "mean_after_abs"

# mean_after_abs, gradual_metric_scale_weight
CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_segm.py --model_name person_car_annot_height_mean_after_abs_gradual_rough${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car --rough_metric_scale_weight ${ROUGH_WEIGHT} --annot_height --rough_metric_loss_func "mean_after_abs" --gradual_metric_scale_weight
