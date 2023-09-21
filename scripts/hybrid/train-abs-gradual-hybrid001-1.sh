ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# batch 8 wo scale dividing and damping_update
FINE_WEIGHT=0.01
ROUGH_WEIGHT=1

# CUDA_VISIBLE_DEVICES=7 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_damping_update_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --damping_update --gradual_metric_scale_weight

# rough: mean_after_abs
# fine: abs / erosion
# others: bs8 / gradual / wo_scale_dividing
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_erosion_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --enable_erosion

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth
# others: bs8 / gradual / wo_scale_dividing
# CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing
# UPDATE_FREQ=3
# CUDA_VISIBLE_DEVICES=4 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing
# UPDATE_FREQ=5
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / gradual_weight_func="sigmoid"
# UPDATE_FREQ=3
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_abs_mean_after_abs_gradual_sigmoid_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --gradual_weight_func "sigmoid" --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update

# # rough: mean_after_abs
# # fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# # others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2
# UPDATE_FREQ=5
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers / gradual_weight_func="sigmoid"
# UPDATE_FREQ=5
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers_abs_mean_after_abs_gradual_sigmoid_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --gradual_weight_func "sigmoid" --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --resume --ckpt_timestamp 06-29-16:15

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers
# UPDATE_FREQ=5
# OUTLIER_RELATIVE_ERROR_TH=0.3
# CUDA_VISIBLE_DEVICES=3 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale
# UPDATE_FREQ=5
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking
# UPDATE_FREQ=5
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=3 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors / with_normal_loss
# UPDATE_FREQ=5
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_normal_loss_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors --with_normal_loss

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors / with_normal_loss
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_normal_loss_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors --with_normal_loss

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors
# UPDATE_FREQ=5
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=1 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:5) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors / with_normal_loss / gradual_normal_weight
# UPDATE_FREQ=5
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=5 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_gradual_normal_loss_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors --with_normal_loss --gradual_normal_weight

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors / with_normal_loss / gradual_normal_weight
# UPDATE_FREQ=3
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=6 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_gradual_normal_loss_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors --with_normal_loss --gradual_normal_weight

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:1) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / soft_remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors
# UPDATE_FREQ=1
# OUTLIER_RELATIVE_ERROR_TH=0.2
# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_soft_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --soft_remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors

# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / soft_remove_outliers0.2 / median_cam_height / median_scale / disable_road_masking / normal_with_8_neighbors
UPDATE_FREQ=3
OUTLIER_RELATIVE_ERROR_TH=0.2
CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq${UPDATE_FREQ}_soft_remove_outliers${OUTLIER_RELATIVE_ERROR_TH}_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --soft_remove_outliers --use_median_cam_height --use_median_scale --disable_road_masking --normal_with_8_neighbors
