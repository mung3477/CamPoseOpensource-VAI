SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}/unimatch:${PYTHONPATH}"

python policy_robosuite/train_implicit_extrinsic.py \
--name train_implicit_extrinsic_use_plucker_debug \
--policy_class dp \
--use_plucker "0" \
--wandb_project_name know_camera_debug \
--wandb_entity DynamicVLA