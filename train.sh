SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}/unimatch:${PYTHONPATH}"
export PYTHONPATH="${REPO_ROOT}/Depth_Anything_V2:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python policy_robosuite/train.py \
--name train_dp_use_plucker_liftrand_eef_delta_use_dynamics_basis_not_overlay_origin_center \
--policy_class dp \
--use_plucker 1 \
--wandb_project_name know_camera_train \
--wandb_entity DynamicVLA \
--num_epochs 80_001 \
--eval_start_epoch 60_000 \
--eval_every 2_000 \
--save_every 2_000 \
--use_dynamics_basis 1 \
--use_overlay_basis 0 \
--use_basis_origin_robot 0

CUDA_VISIBLE_DEVICES=0 python policy_robosuite/train.py \
--name train_dp_use_plucker_liftrand_eef_delta_use_dynamics_basis_not_overlay_origin_robot \
--policy_class dp \
--use_plucker 1 \
--wandb_project_name know_camera_train \
--wandb_entity DynamicVLA \
--num_epochs 80_001 \
--eval_start_epoch 60_000 \
--eval_every 2_000 \
--save_every 2_000 \
--use_dynamics_basis 1 \
--use_overlay_basis 0 \
--use_basis_origin_robot 1

CUDA_VISIBLE_DEVICES=0 python policy_robosuite/train.py \
--name train_dp_no_use_plucker_liftrand_eef_delta_use_dynamics_basis_overlay_origin_center \
--policy_class dp \
--use_plucker 0 \
--wandb_project_name know_camera_train \
--wandb_entity DynamicVLA \
--num_epochs 80_001 \
--eval_start_epoch 60_000 \
--eval_every 2_000 \
--save_every 2_000 \
--use_dynamics_basis 1 \
--use_overlay_basis 1 \
--use_basis_origin_robot 0

CUDA_VISIBLE_DEVICES=0 python policy_robosuite/train.py \
--name train_dp_no_use_plucker_liftrand_eef_delta_use_dynamics_basis_overlay_origin_robot \
--policy_class dp \
--use_plucker 0 \
--wandb_project_name know_camera_train \
--wandb_entity DynamicVLA \
--num_epochs 80_001 \
--eval_start_epoch 60_000 \
--eval_every 2_000 \
--save_every 2_000 \
--use_dynamics_basis 1 \
--use_overlay_basis 1 \
--use_basis_origin_robot 1

# python policy_maniskill/train.py \
# --name train_dp_use_plucker_maniskill \
# --policy_class dp \
# --use_plucker 1 \
# --wandb_project_name know_camera_train \
# --wandb_entity DynamicVLA


