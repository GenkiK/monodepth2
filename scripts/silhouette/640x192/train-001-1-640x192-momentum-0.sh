ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# batch 8 wo scale dividing and damping_update
FINE_WEIGHT=0.01
ROUGH_WEIGHT=1


# rough: mean_after_abs
# fine: abs
# others: momentum / bs8 / gradual / wo_scale_dividing / remove_outliers0.2 / median_cam_height / disable_road_masking / normal_with_8_neighbors / from_ground
OUTLIER_RELATIVE_ERROR_TH=0.2
CUDA_VISIBLE_DEVICES=9 python -OO $ROOT_DIR/train_silhouette_momentum.py --model_name person_car_annot_height-lowres-momentum-bs8-wo_scale_dividing-median_cam_height-remove_outliers${OUTLIER_RELATIVE_ERROR_TH}-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 192 --width 640 --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --gradual_metric_scale_weight --rough_metric_loss_func "mean_after_abs" --remove_outliers --use_median_cam_height --disable_road_masking --normal_with_8_neighbors --from_ground