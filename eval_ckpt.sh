SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}/unimatch:${PYTHONPATH}"
export PYTHONPATH="${REPO_ROOT}/Depth_Anything_V2:${PYTHONPATH}"

EPOCHS=(62000 64000 66000 68000 70000 72000 74000 76000 78000 80000)
GPUS=(0 1 2 3)

max_jobs=4
job_count=0

for i in "${!EPOCHS[@]}"; do
  EPOCH="${EPOCHS[$i]}"
  GPU="${GPUS[$((i % ${#GPUS[@]}))]}"

  echo "[launch] epoch=${EPOCH} -> GPU=${GPU}"

  CUDA_VISIBLE_DEVICES="${GPU}" python policy_robosuite/eval_ckpt.py \
    --name "eval_dp_use_plucker_liftrand_eef_delta_use_dynamics_basis_instead_of_plucker_epoch_${EPOCH}" \
    --policy_class dp \
    --use_plucker 1 \
    --wandb_project_name know_camera_eval \
    --wandb_entity DynamicVLA \
    --use_dynamics_basis 1 \
    --ckpt_dir /data1/local/CamPoseOpensource/policy_robosuite/checkpoints/train_dp_use_plucker_liftrand_eef_delta_use_dynamics_basis_instead_of_plucker \
    --ckpt_path /data1/local/CamPoseOpensource/policy_robosuite/checkpoints/train_dp_use_plucker_liftrand_eef_delta_use_dynamics_basis_instead_of_plucker/epoch_${EPOCH}.pth \
    > "eval_epoch_${EPOCH}.log" 2>&1 &

  ((job_count++))

  # 4개 띄웠으면 다 끝날 때까지 기다렸다가 다음 4개
  if (( job_count % max_jobs == 0 )); then
    wait
  fi
done

# 마지막 배치(4개 미만) 마무리 대기
wait
echo "All evaluations finished."