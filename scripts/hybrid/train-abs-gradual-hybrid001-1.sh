ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

# batch 8 wo scale dividing and damping_update
FINE_WEIGHT=0.01
ROUGH_WEIGHT=1

CUDA_VISIBLE_DEVICES=7 python -OO $ROOT_DIR/train_hybrid.py --model_name person_car_annot_height_bs8_wo_scale_dividing_damping_update_abs_gradual_hybrid${FINE_WEIGHT}_${ROUGH_WEIGHT} --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight $FINE_WEIGHT --rough_metric_scale_weight $ROUGH_WEIGHT --annot_height --learning_rate 5e-5 --gamma 0.5 --cam_height_loss_func "abs" --batch_size 8 --damping_update --gradual_metric_scale_weight