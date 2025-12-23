python policy_robosuite/train.py \
--name train_dp_use_plucker_liftrand_eef_delta \
--policy_class dp \
--use_plucker "1" \
--wandb_project_name know_camera_train \
--wandb_entity DynamicVLA \
--num_epochs 80_001 \
--eval_start_epoch 60_000 \
--eval_every 2_000 \
--save_every 2_000


# python policy_maniskill/train.py \
# --name train_dp_use_plucker_maniskill \
# --policy_class dp \
# --use_plucker 1 \
# --wandb_project_name know_camera_train \
# --wandb_entity DynamicVLA


