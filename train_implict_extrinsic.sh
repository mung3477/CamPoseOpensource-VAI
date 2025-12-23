SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}/unimatch:${PYTHONPATH}"
export PYTHONPATH="${REPO_ROOT}/Depth_Anything_V2:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python policy_robosuite/train_implicit_extrinsic.py \
--name train_implicit_extrinsic_backbone_predict_extrinsic_epoch1000_extrinsic_translation_normalize_use_depth_lr_3e_3 \
--policy_class dp \
--use_plucker "0" \
--wandb_project_name know_camera_implicit_extrinsic_pretrain \
--wandb_entity DynamicVLA \
--use_linear_prob "1" \
--num_dynamic_feature 3 \
--window_size 5 \
--batch_size 256 \
--lr 3e-3 \
--seed 0 \
--num_epochs 1000 \
--use_fp16 "0" \
--use_depth_sim "1"
