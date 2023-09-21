ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# batch 8 wo scale dividing and damping_update
FINE_WEIGHT=0.01
ROUGH_WEIGHT=1


# rough: mean_after_abs
# fine: abs / use_1st_quartile_depth / sparse_update(freq:3) / wo_1st_update
# others: bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / median_scale
UPDATE_FREQ=3
OUTLIER_RELATIVE_ERROR_TH=0.3
CUDA_VISIBLE_DEVICES=2 python -OO $ROOT_DIR/train_hybrid.py --model_name dry-run --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --use_1st_quartile_depth --sparse_update --update_freq ${UPDATE_FREQ} --wo_1st_update --remove_outliers --use_median_cam_height --use_median_scale -n