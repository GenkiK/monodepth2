ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# batch 8 wo scale dividing and damping_update
FINE_WEIGHT=0.01
ROUGH_WEIGHT=0.5

# CUDA_VISIBLE_DEVICES=6 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_damping_update_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --damping_update --gradual_metric_scale_weight

# rough: mean_after_abs
# fine: abs / erosion
# others: bs8 / gradual / wo_scale_dividing
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_erosion_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --enable_erosion

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth
# others: bs8 / gradual / wo_scale_dividing
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing
# UPDATE_FREQ=3
# CUDA_VISIBLE_DEVICES=5 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing
UPDATE_FREQ=5
CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update


# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / gradual_weight_func="sigmoid"
# UPDATE_FREQ=3
# CUDA_VISIBLE_DEVICES=3 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_abs_mean_after_abs_gradual_sigmoid_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --gradual_weight_func "sigmoid" --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update