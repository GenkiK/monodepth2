ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
DEVICE=7

echo ""
echo "script is runnning"
echo ""



CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough05 --ckpt_timestamp "05-10-10:48" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10
CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_with_outliers --ckpt_timestamp "05-10-10:48" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car
CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough01 --ckpt_timestamp "05-10-11:08" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10
CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough1 --ckpt_timestamp "05-10-10:27" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10
CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough2 --ckpt_timestamp "05-10-11:10" --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10

# CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough05 --ckpt_timestamp "05-10-10:48" --height 320 --width 1024 --epoch_for_eval 20 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10
# CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_with_outliers --ckpt_timestamp "05-10-10:48" --height 320 --width 1024 --epoch_for_eval 20 --segm_dirname modified_segms_labels_person_car
# CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough01 --ckpt_timestamp "05-10-11:08" --height 320 --width 1024 --epoch_for_eval 20 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10
# CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough1 --ckpt_timestamp "05-10-10:27" --height 320 --width 1024 --epoch_for_eval 20 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10
# CUDA_VISIBLE_DEVICES=$DEVICE python -OO $ROOT_DIR/partly_evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough2 --ckpt_timestamp "05-10-11:10" --height 320 --width 1024 --epoch_for_eval 20 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10
