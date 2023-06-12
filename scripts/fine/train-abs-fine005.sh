ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# # HACK: this script is for resuming training
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --resume --ckpt_timestamp $CKPT_TIMESTAMP

# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_abs_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --cam_height_loss_func "abs"

# batch 12
# CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_abs_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs"

# bs 8 wo scale divinding
# CUDA_VISIBLE_DEVICES=6 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_bs8_wo_scale_dividing_abs_fine005 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.05 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8

# bs 8 wo scale divinding damping update
CUDA_VISIBLE_DEVICES=4 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_bs8_wo_scale_dividing_damping_update_abs_fine005 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.05 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --damping_update

