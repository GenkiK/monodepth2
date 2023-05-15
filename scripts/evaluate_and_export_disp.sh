ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
DEVICE=0

echo ""
echo "script is runnning"
echo ""



CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_with_outliers --ckpt_timestamp "05-10-10:48" --height 320 --width 1024
CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough01 --ckpt_timestamp "05-10-11:08" --height 320 --width 1024
CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough05 --ckpt_timestamp "05-10-10:48" --height 320 --width 1024
CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough1 --ckpt_timestamp "05-10-10:27" --height 320 --width 1024
CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough2 --ckpt_timestamp "05-10-11:10" --height 320 --width 1024

# HACK:  既に推定して保存した深度を評価していることに注意（--disp_filename_to_evalをつける）
# CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name original --ckpt_timestamp "05-09-17:55" --height 320 --width 1024 --epoch_for_eval 20 --disp_filename_to_eval disps_eigen_split_20.npy
# CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough05 --ckpt_timestamp "05-10-10:48" --height 320 --width 1024 --epoch_for_eval 20 --disp_filename_to_eval disps_eigen_split_20.npy
# CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_with_outliers --ckpt_timestamp "05-10-10:48" --height 320 --width 1024 --epoch_for_eval 20 --disp_filename_to_eval disps_eigen_split_20.npy
# CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough01 --ckpt_timestamp "05-10-11:08" --height 320 --width 1024 --epoch_for_eval 20 --disp_filename_to_eval disps_eigen_split_20.npy
# CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough1 --ckpt_timestamp "05-10-10:27" --height 320 --width 1024 --epoch_for_eval 20 --disp_filename_to_eval disps_eigen_split_20.npy
# CUDA_VISIBLE_DEVICES=DEVICE python -OO $ROOT_DIR/evaluate_and_export_disp.py --model_name person_car_annot_height_offset4_th06_min_inst10_rough2 --ckpt_timestamp "05-10-11:10" --height 320 --width 1024 --epoch_for_eval 20 --disp_filename_to_eval disps_eigen_split_20.npy