ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
DEVICE=6

# CUDA_VISIBLE_DEVICES=$DEVICE python -OO "$ROOT_DIR/export_disp_with_my_model.py" --root_dir $ROOT_DIR --resolution "1024x320" --model_name "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_abs_mean_after_abs_gradual_hybrid0.01_1" --ckpt_timestamp "07-06-17:57"

CUDA_VISIBLE_DEVICES=$DEVICE python -OO "$ROOT_DIR/export_disp_with_my_model.py" --root_dir $ROOT_DIR --resolution "1024x320" --model_name "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1" --ckpt_timestamp "07-18-15:12"