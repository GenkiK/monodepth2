ROOT_DIR="/home/gkinoshita/workspace/monodepth2"
CUDA_VISIBLE_DEVICES=7 python "$ROOT_DIR/export_disp.py" --root_img_dir "$ROOT_DIR/kitti_data" --root_output_dir "$ROOT_DIR/output_disp" --model_type mono --resolution 1024x320