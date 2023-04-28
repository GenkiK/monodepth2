ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
CKPT_TIMESTAMP="04-23-17:56"

# Our standard mono model
CUDA_VISIBLE_DEVICES=7 python $ROOT_DIR/train_with_segm.py --model_name M_1024x320 --height 320 --width 1024 --resume --ckpt_timestamp $CKPT_TIMESTAMP

# train w/o outlier segms
CUDA_VISIBLE_DEVICES=7 python $ROOT_DIR/train_with_segm.py --model_name person_car_annot_height_offset4_th06_min_inst10 --height 320 --width 1024 --segm_dirname modified_segms_labels_person_car_annot_height_offset4_th06_min_inst10

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
