#!bin/bash

# ドライランさせることに注意

ROOT_DIR="../kitti_data"
for i in 0 1; do
    find ../kitti_data -type d -name "image_0$i" -exec rm -rf '{}' +
done;

