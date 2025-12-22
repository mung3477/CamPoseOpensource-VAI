python policy_robosuite/train.py \
--name train_dp_use_plucker_robosuite \
--policy_class dp \
--use_plucker 1 \
--wandb_project_name know_camera_train \
--wandb_entity DynamicVLA

python policy_maniskill/train.py \
--name train_dp_use_plucker_maniskill \
--policy_class dp \
--use_plucker 1 \
--wandb_project_name know_camera_train \
--wandb_entity DynamicVLA


