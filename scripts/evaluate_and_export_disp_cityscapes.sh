ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
DEVICE=1
EPOCH=19

echo ""
echo "script is runnning"
echo ""

CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp_cityscapes.py \
    --model_name vadepth --ckpt_timestamp "06-30-12:35" --height 320 --width 1024 --epoch_for_eval $EPOCH \
    --data_path '/home/gkinoshita/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test' \
    --enable_loading_disp_to_eval

CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp_cityscapes.py \
    --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq5_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1 \
    --ckpt_timestamp "07-18-15:11" --height 320 --width 1024 --epoch_for_eval $EPOCH \
    --data_path '/home/gkinoshita/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test'

CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp_cityscapes.py \
    --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1 \
    --ckpt_timestamp "07-18-15:12" --height 320 --width 1024 --epoch_for_eval $EPOCH \
    --data_path '/home/gkinoshita/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test'


#### epoch=last epoch

CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp_cityscapes.py --model_name vadepth --ckpt_timestamp "06-30-12:35" --height 320 --width 1024 --data_path '/home/gkinoshita/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test' --enable_loading_disp_to_eval
CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp_cityscapes.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq5_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1 --ckpt_timestamp "07-18-15:11" --height 320 --width 1024 --data_path '/home/gkinoshita/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test'
CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp_cityscapes.py --model_name person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1 --ckpt_timestamp "07-18-15:12" --height 320 --width 1024 --data_path '/home/gkinoshita/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test'