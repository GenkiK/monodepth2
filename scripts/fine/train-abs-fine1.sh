ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# disable_automasking
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_abs_disable_mask_fine1 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 1.0 --annot_height --cam_height_loss_func "abs" --disable_automasking

# warmup
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_abs_warmup_fine1 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 1.0 --annot_height --cam_height_loss_func "abs" --warmup

CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_abs_fine1 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 1.0 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs"
