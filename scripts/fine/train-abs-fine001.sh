ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
CKPT_TIMESTAMP="05-23-18:09"

echo "script is runnning"
echo ""

# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --resume --ckpt_timestamp $CKPT_TIMESTAMP

# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_abs_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --cam_height_loss_func "abs"

# batch 12
# CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_abs_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs"

# batch 8 wo scale dividing
# CUDA_VISIBLE_DEVICES=7 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_bs8_wo_scale_dividing_abs_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8

# batch 8 wo scale dividing and damping_update
# CUDA_VISIBLE_DEVICES=8 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_bs8_wo_scale_dividing_damping_update_abs_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --damping_update

# batch 8 wo scale dividing and damping_update and
CUDA_VISIBLE_DEVICES=8 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_bs8_wo_scale_dividing_damping_update_init_after_1st_epoch_abs_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --damping_update --init_after_1st_epoch --log_dirname_1st_epoch "person_car_annot_height_bs8_wo_scale_dividing_damping_update_abs_fine001_06-08-14:29"
