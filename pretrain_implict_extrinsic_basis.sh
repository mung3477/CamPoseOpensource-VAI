SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}/unimatch:${PYTHONPATH}"
export PYTHONPATH="${REPO_ROOT}/Depth_Anything_V2:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python policy_robosuite/pretrain_implicit_extrinsic_basis.py \
--name train_backbone_pred_basis_ep1000_lr_3e_3_dynamic_action_x1_y1_z1_with_conv__debug_all_1_0_debug \
--policy_class dp \
--use_plucker "0" \
--wandb_project_name know_camera_implicit_extrinsic_pretrain_debug \
--wandb_entity DynamicVLA \
--use_linear_prob "1" \
--num_dynamic_feature 3 \
--window_size 5 \
--batch_size 2 \
--lr 3e-3 \
--seed 0 \
--num_epochs 1000 \
--use_fp16 "0" \
--translation_normalize_extrinsic "1" \
--use_depth_sim "0" \
--use_robot_eef_poses "0" \
--pred_basis "1" \
--debug "1"
