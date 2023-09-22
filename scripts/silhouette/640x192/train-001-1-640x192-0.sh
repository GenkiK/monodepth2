ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# batch 8 wo scale dividing and damping_update
FINE_WEIGHT=0.01
ROUGH_WEIGHT=1


# # rough: mean_after_abs
# # fine: abs / sparse_update(freq:3) / wo_1st_update
# # others: bs8 / gradual / wo_scale_dividing / remove_outliers0.3 / median_cam_height / disable_road_masking / normal_with_8_neighbors
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.3
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_silhouette.py --model_name person_car_annot_height-lowres-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-update_freq${UPDATE_FREQ}-remove_outliers${OUTLIER_RELATIVE_ERROR_TH}-disable_road_masking-8neighbors-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs"  --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --disable_road_masking --normal_with_8_neighbors

# # rough: mean_after_abs
# # fine: abs / sparse_update(freq:3) / wo_1st_update
# # others: bs8 / gradual / wo_scale_dividing / remove_outliers0.3 / median_cam_height / disable_road_masking / normal_with_8_neighbors / from_ground
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.3
# CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_silhouette.py --model_name person_car_annot_height-lowres-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-update_freq${UPDATE_FREQ}-remove_outliers${OUTLIER_RELATIVE_ERROR_TH}-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs"  --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --disable_road_masking --normal_with_8_neighbors --from_ground

# # rough: mean_after_abs
# # fine: abs / wo_1st_update
# # others: bs8 / gradual / wo_scale_dividing / median_cam_height / disable_road_masking / normal_with_8_neighbors / from_ground
# CUDA_VISIBLE_DEVICES=3 python -OO $ROOT_DIR/train_silhouette.py --model_name person_car_annot_height-lowres-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --wo_1st_update --use_median_cam_height --disable_road_masking --normal_with_8_neighbors --from_ground

# # rough: mean_after_abs
# # fine: abs / sparse_update(freq:3) / wo_1st_update
# # others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / disable_road_masking / normal_with_8_neighbors / from_ground
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=4 python -OO $ROOT_DIR/train_silhouette.py --model_name person_car_annot_height-lowres-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-update_freq${UPDATE_FREQ}-remove_outliers${OUTLIER_RELATIVE_ERROR_TH}-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs"  --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --disable_road_masking --normal_with_8_neighbors --from_ground

# rough: mean_after_abs
# fine: abs / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / disable_road_masking / normal_with_8_neighbors / from_ground
OUTLIER_RELATIVE_ERROR_TH=0.2
CUDA_VISIBLE_DEVICES=4 python -OO $ROOT_DIR/train_silhouette.py --model_name person_car_annot_height-lowres-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-remove_outliers${OUTLIER_RELATIVE_ERROR_TH}-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --wo_1st_update --remove_outliers --use_median_cam_height --disable_road_masking --normal_with_8_neighbors --from_ground

# # rough: mean_after_abs
# # fine: abs / sparse_update(freq:2) / wo_1st_update
# # others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / disable_road_masking / normal_with_8_neighbors / from_ground
# UPDATE_FREQ=2
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=6 python -OO $ROOT_DIR/train_silhouette.py --model_name person_car_annot_height-lowres-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-update_freq${UPDATE_FREQ}-remove_outliers${OUTLIER_RELATIVE_ERROR_TH}-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs"  --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --disable_road_masking --normal_with_8_neighbors --from_ground

# # rough: mean_after_abs
# # fine: abs / sparse_update(freq:2) / wo_1st_update
# # others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / disable_road_masking / normal_with_8_neighbors / from_ground
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=6 python -OO $ROOT_DIR/train_silhouette.py --model_name person_car_annot_height-lowres-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-update_freq${UPDATE_FREQ}-remove_outliers${OUTLIER_RELATIVE_ERROR_TH}-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs"  --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --disable_road_masking --normal_with_8_neighbors --from_ground