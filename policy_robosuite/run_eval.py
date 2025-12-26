import h5py
import json
import os
import argparse
from pathlib import Path

import torch

import robosuite as suite
from robosuite.wrappers.action_wrapper import wrap_env_action_space

from utils import set_seed, load_data, get_all_ckpts

from eval import Evaluator
from models.dp import DiffusionPolicy

torch.backends.cuda.enable_flash_sdp(True)
print(torch.backends.cuda.is_flash_attention_available())

def argparse_eval():
	parser = argparse.ArgumentParser(description="Train ACT policy for robosuite environments")
	def str2bool(v): return str(v) == "1"

	# General config
	parser.add_argument('--name', type=str, default='oct16_open_source', help='name for the run')
	parser.add_argument('--dataset_dir', type=str, default=None,
						help='Path to dataset directory (absolute). If None, defaults to policy_robosuite/demos')
	parser.add_argument('--dataset_suffix', type=str, default='liftrand_eef_delta')
	parser.add_argument('--dataset_path', type=str, default=None)
	parser.add_argument('--camera_poses_dir', type=str, default=None,
						help='Path to camera poses directory (absolute). If None, defaults to policy_robosuite/camera_poses')
	parser.add_argument('--ckpt_dir', type=str, default=None,
						help='Path to checkpoints directory (absolute). If None, defaults to policy_robosuite/checkpoints/<name>')
	parser.add_argument('--policy_class', type=str, default='act', choices=['dp','act','smolvla'], help='policy class')

	parser.add_argument('--num_episodes', default=200, type=int, help='num_episodes')
	parser.add_argument('--use_plucker', default=False, type=str2bool, help='use Plucker embeddings')

	# Camera pose config
	parser.add_argument('--train_poses_file', type=str, default='train_cameras.json', help='Path to training camera poses JSON (old flat format)')
	parser.add_argument('--test_poses_file', type=str, default='test_cameras.json', help='Path to test camera poses JSON (old flat format)')
	parser.add_argument('--pose_files', nargs='+', default=['train_cameras.json', 'test_cameras.json'], help='Pose files to evaluate on (default: test only)')
	parser.add_argument('--n', type=int, default=3, help='Number of cameras per window W(i) = [m*i, m*i + n)')
	parser.add_argument('--m', type=int, default=1, help='Stride for camera window W(i) = [m*i, m*i + n)')
	parser.add_argument('--num_side_cam', type=int, default=1, choices=[1,2], help='Number of side cams to use (1 or 2)')
	parser.add_argument('--default_cam', type=str2bool, default=False, help='When true, use default agentview pose for all cameras (duplicate if >1)')

	parser.add_argument('--batch_size', default=70, type=int, help='batch_size')
	parser.add_argument('--seed', default=0, type=int, help='seed')
	parser.add_argument('--num_epochs', default=30_001, type=int, help='num_epochs')
	parser.add_argument('--eval_start_epoch', type=int, default=20_000, help='start evaluating 50 at this epoch')
	parser.add_argument('--lr', type=float, default=2e-5, help='lr')
	parser.add_argument('--save_every', type=int, default=1000, help='save checkpoint every N epochs')
	parser.add_argument('--use_fp16', default=True, type=str2bool, help='use mixed precision bf16 training')

	# Dataloader config
	parser.add_argument('--transform', type=str, default='crop', choices=['crop', 'id', 'crop_jitter'],
						help='Image transformation type')
	parser.add_argument('--prob_drop_proprio', default=1., type=float, help='probability to drop proprio')
	parser.add_argument('--use_cam_pose', default=True, type=bool, help='otherwise mask to 0')
	parser.add_argument('--original', default=False, type=str2bool, help='visually same as original lift')

	# ACT model config
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
	parser.add_argument('--latent_drop_prob', type=float, default=0.0, help='drop probability for RGB latents in backbone')
	parser.add_argument('--kl_weight', type=float, default=1.0, help='KL Weight')
	parser.add_argument('--chunk_size', type=int, default=30, help='chunk_size')
	parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
	parser.add_argument('--obs_dim', type=int, default=7, help='observation dimension')


	parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
	parser.add_argument('--ffn_dim', type=int, default=2048, help='feedforward network dimension')
	parser.add_argument('--enc_layers', type=int, default=4, help='number of encoder layers')
	parser.add_argument('--dec_layers', type=int, default=7, help='number of decoder layers')
	parser.add_argument('--pre_norm', type=bool, default=True, help='use pre-normalization')
	parser.add_argument('--activation', default='relu', help='activation function')

	# Backbone config
	parser.add_argument('--backbone', default='late_imagenet', help='backbone: resnet, linear')
	parser.add_argument('--patch_size', type=int, default=8, help='patch size')

	# Evaluation config
	parser.add_argument('--eval_episodes', type=int, default=50, help='number of evaluation episodes')
	parser.add_argument('--eval_max_steps', type=int, default=300, help='max steps per evaluation episode')
	parser.add_argument('--eval_save_n_video', type=int, default=10, help='save the first n videos for each evaluation epoch')

	args = parser.parse_args()

	# Anchor paths under the module root (policy_robosuite)
	MODULE_ROOT = Path(__file__).resolve().parent

	# dataset_dir -> dataset_path
	if args.dataset_dir is None:
		args.dataset_dir = str((MODULE_ROOT / "demos").resolve())
	args.dataset_path = os.path.join(args.dataset_dir, args.dataset_suffix + ".hdf5")

	# camera poses dir
	if args.camera_poses_dir is None:
		args.camera_poses_dir = str((MODULE_ROOT / "camera_poses").resolve())
	if not hasattr(args, 'ckpt_dir') or args.ckpt_dir is None:
		args.ckpt_dir = str((MODULE_ROOT / "checkpoints" / args.name).resolve())
	os.makedirs(args.ckpt_dir, exist_ok=True)

	# save eval config
	config_path = os.path.join(args.ckpt_dir, f'{args.name}_eval_config.json')
	with open(config_path, 'w') as f:
		json.dump(vars(args), f, indent=4)

	return args

def build_env(args):
	# Build environment from dataset metadata
	with h5py.File(args.dataset_path, 'r') as f:
		env_args_raw = f['data'].attrs['env_args']
		action_space = f['data'].attrs['action_space']
	env_config = json.loads(env_args_raw)
	env_name = env_config['env_name']
	env_kwargs = env_config['env_kwargs']

	env_kwargs.update({
		'has_renderer': False,
		'has_offscreen_renderer': True,
		'use_camera_obs': False,
		'camera_heights': 256,
		'camera_widths': 256,
	})

	if args.original and 'Rand' in env_name:
		env_kwargs['original'] = True

	env = suite.make(env_name=env_name, camera_names=["agentview"], **env_kwargs)
	if action_space in ('eef_delta', 'joint_delta'):
		env = wrap_env_action_space(env, action_space)
	env.reset()

	return env

def load_state_dict(ckpt, policy):
	policy.load_state_dict(ckpt['model_state_dict'])

	epoch = ckpt['epoch']
	print(f"Evaluate checkpoint at epoch {ckpt['epoch']}")

	return epoch

def setup_eval_dir(pose_file: str, ckpt_dir: str, epoch: int):
	pose_name = pose_file[:-5]  # Remove .json extension
	eval_save_path = os.path.join(ckpt_dir, f"eval_epoch_{epoch}_{pose_name}")
	os.makedirs(eval_save_path, exist_ok=True)

	return pose_name, eval_save_path

def run_eval(policy, ckpt, env, args, evaluator):
	assert ckpt is not None, "Checkpoint should be provided"
	epoch = load_state_dict(ckpt, policy)
	policy.eval()

	# IMPORTANT: Import after suite.make()
	import OpenGL.GL as gl

	# Check OpenGL framebuffer. This is a hack to fix the renderer issue in robosuite.
	if gl.GL_FRAMEBUFFER_COMPLETE != gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER):
		print("⚠️  Render fault detected; rebuilding context")
		env.reset()

	for pose_file in args.pose_files:
		pose_name, eval_save_path = setup_eval_dir(
			pose_file,
			ckpt_dir = args.ckpt_dir,
			epoch = epoch
		)

		evaluator.success_by_seed = {}

		success_rates = []
		for episode_idx in range(args.eval_episodes):
			with torch.no_grad():
				_, success_rate, _ = evaluator.evaluate(
					policy=policy,
					save_path=eval_save_path,
					video_prefix=f"epoch_{epoch}_episode_{episode_idx}",
					pose_name=pose_name,
					episode_num=episode_idx
				)
			success_rates.append(success_rate)
		avg_success_rate = sum(success_rates) / len(success_rates)
		print(f'success_rate_{pose_name}: {avg_success_rate}, step={epoch}')

		with open(os.path.join(eval_save_path, 'success_by_seed.json'), 'w') as f:
			json.dump(evaluator.success_by_seed, f, indent=2)

def evaluate(args):
	ckpt_paths = get_all_ckpts(args.ckpt_dir)
	assert ckpt_paths is not None, f'{args.ckpt_dir} does not contain any *.pth files!'
	print(f"Evaluate following checkpoints: {ckpt_paths}")

	set_seed(args.seed)
	args.action_dim = 8 if 'joint' in args.dataset_path else 7

	env = build_env(args)

	# Even dataloader is not used, assigning return values to arbitrary variables are needed
	# to keep env.sim intact.
	train_dataloader, val_dataloader, stats = load_data(
		args=args,
		env=env
	)

	evaluator = Evaluator(env=env, norm_stats=stats, dataset_path=args.dataset_path, args=args)

	assert args.policy_class == 'dp', 'We currently test only Diffusion Policy models'
	policy = DiffusionPolicy(args).cuda()

	for ckpt_path in ckpt_paths:
		print(f"Evaluating checkpoint: {ckpt_path}")
		ckpt = torch.load(ckpt_path)
		run_eval(policy, ckpt, env, args, evaluator)

if __name__ == "__main__":
	args = argparse_eval()
	evaluate(args)
