# Define the path to your experiment folder
EXP_DIR="/root/Desktop/workspace/CamPoseOpensource-VAI/policy_robosuite/checkpoints/train_dp_no_plucker_liftrand_eef_delta"

# This loop finds eval folders, extracts the epoch number,
# and zips the .json files only if the .pth file exists.
for eval_dir in "$EXP_DIR"/eval_epoch_*_*/; do
    # 1. Extract the epoch number (e.g., 80000) from the directory name
    EPOCH=$(echo "$(basename "$eval_dir")" | grep -oP 'epoch_\K\d+')

    # 2. Check if the parent directory contains epoch_[EPOCH].pth
    if [ -f "$EXP_DIR/epoch_$EPOCH.pth" ]; then
        echo "Adding JSONs for Epoch $EPOCH..."
        zip -r filtered_jsons.zip "$eval_dir" -i "*.json"
    fi
done

# Also add the main config files from the root (since it has checkpoints)
zip filtered_jsons.zip "$EXP_DIR"/*.json
