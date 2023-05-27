ROOT_DIR="/home/gkinoshita/workspace/monodepth2"

# train w/o outlier segms
# # HACK: this script is for resuming training
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/batch_calc_height_debug.py

# CUDA_VISIBLE_DEVICES=0 python -OO $ROOT_DIR/train_with_road.py --model_name person_car_annot_height_fine001 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_road --fine_metric_scale_weight 0.01 --annot_height


# # Our low resolution mono model
# python ../train.py --model_name M_416x128 \
#   --height 128 --width 416

# # Our high resolution mono model
# python ../train.py --model_name M_1024x320 \
#   --height 320 --width 1024 \
#   --load_weights_folder ~/tmp/M_640x192/models/weights_9 \
#   --num_epochs 5 --learning_rate 1e-5

# # Our standard mono model w/o pretraining
# python ../train.py --model_name M_640x192_no_pt \
#   --weights_init scratch \
#   --num_epochs 30

# # Baseline mono model, i.e. ours with our contributions turned off
# python ../train.py --model_name M_640x192_baseline \
#   --v1_multiscale --disable_automasking --avg_reprojection

# # Mono without full-res multiscale
# python ../train.py --model_name M_640x192_no_full_res_ms \
#   --v1_multiscale

# # Mono without automasking
# python ../train.py --model_name M_640x192_no_automasking \
#   --disable_automasking

# # Mono without min reproj
# python ../train.py --model_name M_640x192_no_min_reproj \
#   --avg_reprojection

# # Mono with Zhou's masking scheme instead of ours
# python ../train.py --model_name M_640x192_zhou_masking \
#   --disable_automasking --zhou_mask
