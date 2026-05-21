# mkdir -p /datasets/ssv2_libero_90/depth_train_gt

# for d in /datasets/ssv2/nips/depth_train/*; do
#   [ -d "$d" ] || continue
#   name=$(basename "$d")

#   if [ ! -e "/datasets/ssv2_libero_90/depth_train_gt/$name" ]; then
#     ln -s "$d" "/datasets/ssv2_libero_90/depth_train_gt/$name"
#   fi
# done

mkdir /datasets/ssv2_libero_90/depth_train_gt/

# ln -s /datasets/ssv2/nips/depth_train/* /datasets/ssv2_libero_90/depth_train_gt/
ln -s /datasets/libero_corl/libero-90_depth_ssv2_gt/* /datasets/ssv2_libero_90/depth_train_gt/

