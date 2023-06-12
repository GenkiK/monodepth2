ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

echo "script is runnning"
echo ""

CUDA_VISIBLE_DEVICES=4 python -OO $ROOT_DIR/train_with_segm.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough2 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10 --rough_metric_scale_weight 2.0
