import os

import json
import time
import torch
import argparse
import tqdm
import robosuite as suite
from robosuite.wrappers.action_wrapper import wrap_env_action_space
from contextlib import nullcontext
import h5py
from pathlib import Path

from utils import load_data, load_implicit_extrinsic_data, compute_dict_mean, set_seed, detach_dict, constant_schedule, cleanup_ckpt, get_last_ckpt, cosine_schedule_with_warmup
from models.act import ACTPolicy
from eval import Evaluator
from models.dp import DiffusionPolicy
from models.smolvla import SmolVLAPolicyWrapper
from unimatch.unimatch import UniMatch, UniMatchFlowWDepth
import einops

import wandb

torch.backends.cuda.enable_flash_sdp(True)
print(torch.backends.cuda.is_flash_attention_available())

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- utils: 6D rotation -> rotation matrix (Zhou et al.) ----------
def rot6d_to_matrix(x6: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]

    b1 = F.normalize(a1, dim=-1, eps=eps)

    # a2에서 b1 성분 제거
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    a2_ortho = a2 - proj * b1

    b2 = F.normalize(a2_ortho, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)
    return R

def geodesic_rot_loss(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    R_pred, R_gt: (B, 3, 3)
    returns mean angle (radians)
    """
    # relative rotation
    R_rel = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (B,)
    return theta.mean()

def make_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    R: (B,3,3), t: (B,3)
    return: (B,4,4)
    """
    B = R.shape[0]
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T

optical_backbone_cfg = {
    "feature_channels":128,
    "num_scales":2,
    "upsample_factor":4,
    "num_head":1,
    "ffn_dim_expansion":4,
    "num_transformer_layers":6,
    "reg_refine":True,
    "task":'flow',
    "resume_train": "unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
    "strict_resume": False,
}

def load_unimatch_backbone(device, use_dynamic_common_feature=True, num_dynamic_feature=3, use_linear_prob=True, load_pretrained_dynamic_model_path=None, use_depth=False, use_robot_eef_poses=False, pred_basis=False):
    #define optical backbone class
    optical_backbone = UniMatch(feature_channels=optical_backbone_cfg["feature_channels"],
                    num_scales=optical_backbone_cfg["num_scales"],
                    upsample_factor=optical_backbone_cfg["upsample_factor"],
                    num_head=optical_backbone_cfg["num_head"],
                    ffn_dim_expansion=optical_backbone_cfg["ffn_dim_expansion"],
                    num_transformer_layers=optical_backbone_cfg["num_transformer_layers"],
                    reg_refine=optical_backbone_cfg["reg_refine"],
                    task=optical_backbone_cfg["task"],
                    ).to(device)
    optical_backbone_cfg["resume"] = optical_backbone_cfg["resume_train"]
    if optical_backbone_cfg["resume"]:
        print('Load Flow checkpoint: %s' % optical_backbone_cfg["resume"])
        optical_checkpoint = torch.load(optical_backbone_cfg["resume"])
        optical_backbone.load_state_dict(optical_checkpoint['model'], strict=optical_backbone_cfg["strict_resume"])
    backbone_projector_model = UniMatchFlowWDepth(optical_backbone=optical_backbone, use_dynamic_common_feature=use_dynamic_common_feature, num_dynamic_feature=num_dynamic_feature, use_linear_prob=use_linear_prob, load_pretrained_dynamic_model_path=load_pretrained_dynamic_model_path, use_depth=use_depth, use_robot_eef_poses=use_robot_eef_poses, pred_basis=pred_basis)
    #backbone_projector_model = UniMatchVisionBackbone(base_unimatch=backbone_model, fuse_multiscale=False, use_dynamic_common_feature=use_dynamic_common_feature, num_dynamic_feature=num_dynamic_feature, use_linear_prob=use_linear_prob)
    return backbone_projector_model


def main(args, ckpt=None):
    set_seed(args.seed)
    start_time = time.time()

    args.action_dim = 8 if 'joint' in args.dataset_path else 7

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
    
    # IMPORTANT: Import after suite.make()
    import OpenGL.GL as gl

    implicit_extrinsic_backbone = load_unimatch_backbone(device="cuda", use_dynamic_common_feature=True, num_dynamic_feature=args.num_dynamic_feature, use_linear_prob=args.use_linear_prob, load_pretrained_dynamic_model_path=args.load_pretrained_dynamic_model_path, use_depth=args.use_depth_model or args.use_depth_sim, use_robot_eef_poses=args.use_robot_eef_poses, pred_basis=args.pred_basis).to(torch.device("cuda"))
    lr = getattr(args, "lr", 3e-4)
    wd = getattr(args, "weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(implicit_extrinsic_backbone.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))

    train_dataloader, val_dataloader, stats = load_implicit_extrinsic_data(
        args=args,
        env=env,
        preprocess_model=implicit_extrinsic_backbone,
    )
    # Save stats
    os.makedirs(args.ckpt_dir, exist_ok=True)
    stats_path = os.path.join(args.ckpt_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=4)
    # ---- scheduler (warmup + cosine) ----
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(getattr(args, "warmup_ratio", 0.05) * total_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay to min_lr_ratio
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = getattr(args, "min_lr_ratio", 0.1)  # end at 10% of lr
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)    
    epoch = 0
    if ckpt is not None:
        implicit_extrinsic_backbone.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch = ckpt['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']}")
    
    pbar = tqdm.tqdm(total=args.num_epochs, desc="Training")
    pbar.update(epoch)
    while epoch < args.num_epochs:
        
        # Check OpenGL framebuffer. This is a hack to fix the renderer issue in robosuite.
        if gl.GL_FRAMEBUFFER_COMPLETE != gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER):
            print("⚠️  Render fault detected; rebuilding context")
            env.reset()
        # Validation
        if epoch % 10 == 0:
            with torch.inference_mode():
                implicit_extrinsic_backbone.eval()
                epoch_dicts = []
                import pdb; pdb.set_trace()
                for data in val_dataloader:
                    import pdb; pdb.set_trace()
                    with torch.autocast("cuda", dtype=torch.bfloat16) if args.use_fp16 else nullcontext():
                        optical_flow = data['optical_flow']
                        action = data['dynamic_actions_normalized']
                        gt_extrinsic = data['cam_extrinsics'][:, 0]
                        optical_flow = einops.rearrange(optical_flow, "b s n ... -> s b n ...")
                        action = einops.rearrange(action, "b s n ... -> s b n ...")
                        optical_flow = einops.rearrange(optical_flow, "s b c h w -> (s b) c h w")
                        flat_action = einops.rearrange(action, "s b a -> (s b) a")

                        if args.use_robot_eef_poses:
                            robot_eef_poses = data['robot_eef_poses'].squeeze(2)
                            robot_eef_poses = einops.rearrange(robot_eef_poses, "b s n ... -> s b n ...")
                            robot_eef_poses = einops.rearrange(robot_eef_poses, "s b n ... -> (s b) n ...")
                        else:
                            robot_eef_poses = None
                        preds = implicit_extrinsic_backbone(action=flat_action, pre_extract_flow=optical_flow, robot_eef_poses=robot_eef_poses)
                        preds = implicit_extrinsic_backbone(action=flat_action, pre_extract_flow=optical_flow, robot_eef_poses=robot_eef_poses)
                        bp = preds.view(preds.shape[0], 3, 2).float()
                        gt_basis = data["motion_dynamics_basis_tensor"][:, 0]    # (B,3,2) expected
                        eps = 1e-6
                        bg = gt_basis.float()

                        bp = bp / (bp.norm(dim=-1, keepdim=True) + eps)
                        bg = bg / (bg.norm(dim=-1, keepdim=True) + eps)
                        cos_sim = (bp * bg).sum(dim=-1)         # (B,3)
                        loss = (1.0 - cos_sim).mean()

                        forward_dict = {
                            "loss": loss,
                            "loss_basis": loss.detach(),
                            "basis_cos": cos_sim.mean().detach(),
                        }
                        # t_pred = preds[:, 0:3]
                        # r6_pred = preds[:, 3:9]

                        # R_pred = rot6d_to_matrix(r6_pred, eps=1e-6)          # float32
                        # R_gt   = gt_extrinsic[:, :3, :3]
                        # t_gt   = gt_extrinsic[:, :3, 3]

                        # loss_t = F.smooth_l1_loss(t_pred, t_gt)
                        # loss_R = F.mse_loss(R_pred, R_gt)               # <- geodesic 대신 L2
                        # # loss_R = geodesic_rot_loss(R_pred, R_gt)            # float32
                        # loss = loss_t + loss_R
                        # with torch.no_grad():
                        #     rot_err_rad = geodesic_rot_loss(R_pred.float(), R_gt.float())
                        #     rot_err_deg = rot_err_rad.detach() * (180.0 / math.pi)
                        # # rot_err_deg = (loss_R.detach() * (180.0 / math.pi))
                        # trans_err = ((t_pred - t_gt) * stats['val_t_scale'][0]).norm(dim=-1).mean()
                        # forward_dict = {
                        #     "loss": loss,
                        #     "loss_t": loss_t.detach(),
                        #     "loss_R": loss_R.detach(),
                        #     "rot_err_deg": rot_err_deg,
                        #     "trans_err": trans_err,
                        # }
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                for k, v in epoch_summary.items():
                    wandb.log({f'val_{k}': v.item()}, step=epoch)

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch, 
                'model_state_dict': implicit_extrinsic_backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_summary['loss'], 
                'wandb_id': wandb.run.id
            }, checkpoint_path)
            cleanup_ckpt(args.ckpt_dir, keep=3)  # Keep last 3 checkpoints

            if time.time() - start_time > 7.5 * 60 * 60:
                print(f"⏰ Time limit reached ({(time.time() - start_time)/3600:.1f} hours). Exiting...")
                break

        # Check OpenGL framebuffer. This is a hack to fix the renderer issue in robosuite.
        if gl.GL_FRAMEBUFFER_COMPLETE != gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER):
            print("⚠️  Render fault detected; rebuilding context")
            env.reset()

        # Training
        train_history = []
        implicit_extrinsic_backbone.train()
        for data in train_dataloader:

            with torch.autocast("cuda", dtype=torch.bfloat16) if args.use_fp16 else nullcontext():
                # img1 = data['image']
                # img2 = data['future_image']
                optical_flow = data['optical_flow']
                action = data['dynamic_actions_normalized']
                gt_extrinsic = data['cam_extrinsics'][:, 0]
                optical_flow = einops.rearrange(optical_flow, "b s n ... -> s b n ...")
                action = einops.rearrange(action, "b s n ... -> s b n ...")
                optical_flow = einops.rearrange(optical_flow, "s b c h w -> (s b) c h w")
                flat_action = einops.rearrange(action, "s b a -> (s b) a")
                if args.use_robot_eef_poses:
                    robot_eef_poses = data['robot_eef_poses'].squeeze(2)
                    robot_eef_poses = einops.rearrange(robot_eef_poses, "b s n ... -> s b n ...")
                    robot_eef_poses = einops.rearrange(robot_eef_poses, "s b n ... -> (s b) n ...")
                else:
                    robot_eef_poses = None
                preds = implicit_extrinsic_backbone(action=flat_action, pre_extract_flow=optical_flow, robot_eef_poses=robot_eef_poses)
                bp = preds.view(preds.shape[0], 3, 2).float()
                gt_basis = data["motion_dynamics_basis_tensor"][:, 0]    # (B,3,2) expected
                eps = 1e-6
                bg = gt_basis.float()

                bp = bp / (bp.norm(dim=-1, keepdim=True) + eps)
                bg = bg / (bg.norm(dim=-1, keepdim=True) + eps)

                cos_sim = (bp * bg).sum(dim=-1)         # (B,3)
                loss = (1.0 - cos_sim).mean()

                forward_dict = {
                    "loss": loss,
                    "loss_basis": loss.detach(),
                    "basis_cos": cos_sim.mean().detach(),
                }
                # t_pred = preds[:, 0:3]
                # r6_pred = preds[:, 3:9]

                # R_pred = rot6d_to_matrix(r6_pred, eps=1e-6)          # float32
                # R_gt   = gt_extrinsic[:, :3, :3]
                # t_gt   = gt_extrinsic[:, :3, 3]

                # loss_t = F.smooth_l1_loss(t_pred, t_gt)
                # loss_R = F.mse_loss(R_pred, R_gt)               # <- geodesic 대신 L2
                # # loss_R = geodesic_rot_loss(R_pred, R_gt)            # float32
                # loss = loss_t + loss_R
                # with torch.no_grad():
                #     rot_err_rad = geodesic_rot_loss(R_pred.float(), R_gt.float())
                #     rot_err_deg = rot_err_rad.detach() * (180.0 / math.pi)
                # # rot_err_deg = (loss_R.detach() * (180.0 / math.pi))
                # trans_err = ((t_pred - t_gt) * stats['val_t_scale'][0]).norm(dim=-1).mean()
                # forward_dict = {
                #     "loss": loss,
                #     "loss_t": loss_t.detach(),
                #     "loss_R": loss_R.detach(),
                #     "rot_err_deg": rot_err_deg,
                #     "trans_err": trans_err,
                # }
            loss.backward()
            torch.nn.utils.clip_grad_norm_(implicit_extrinsic_backbone.parameters(), max_norm=1.0)
            optimizer.step()
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            train_history.append(detach_dict(forward_dict))
        
        epoch_summary = compute_dict_mean(train_history)
        for k, v in epoch_summary.items():
            wandb.log({f'train_{k}': v.item()}, step=epoch)

        pbar.update(1)
        epoch += 1

    pbar.close()
    env.close()
    wandb.finish()

if __name__ == '__main__':
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

    # Training config
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--num_epochs', default=120, type=int, help='num_epochs')
    parser.add_argument('--eval_start_epoch', type=int, default=20_000, help='start evaluating 50 at this epoch')
    parser.add_argument('--lr', type=float, default=2e-5, help='lr')
    parser.add_argument('--save_every', type=int, default=10, help='save checkpoint every N epochs')
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
    parser.add_argument('--eval_every', type=int, default=1000, help='evaluate every N epochs')
    parser.add_argument('--eval_episodes', type=int, default=10, help='number of evaluation episodes')
    parser.add_argument('--eval_max_steps', type=int, default=300, help='max steps per evaluation episode')
    parser.add_argument('--eval_save_n_video', type=int, default=10, help='save the first n videos for each evaluation epoch')

    # SmolVLA finetuning flags
    parser.add_argument('--freeze_vision_encoder', type=str2bool, default=False, help='freeze the vision encoder (SigLIP)')
    parser.add_argument('--train_expert_only', type=str2bool, default=False, help='train only the action expert; freeze VLM')
    parser.add_argument('--wandb_project_name', type=str, default='DynamicVLA', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')

    parser.add_argument('--num_dynamic_feature', type=int, default=3, help='number of dynamic action features to use')
    parser.add_argument('--window_size', type=int, default=5, help='window size for temporal context')
    parser.add_argument('--use_linear_prob', default=False, type=str2bool, help='use linear probability')
    parser.add_argument('--load_pretrained_dynamic_model_path', type=str, default=None, help='path to pretrained dynamic model checkpoint')
    parser.add_argument('--translation_normalize_extrinsic', default=False, type=str2bool, help='whether to normalize the extrinsic translation')
    parser.add_argument('--use_depth_sim', default=False, type=str2bool, help='use depth input in backbone')
    parser.add_argument('--use_depth_model', default=False, type=str2bool, help='use depth input in backbone')
    parser.add_argument('--use_robot_eef_poses', default=False, type=str2bool, help='use robot eef position in backbone')
    parser.add_argument('--pred_basis', default=False, type=str2bool, help='predict basis vectors instead of rotation matrix directly')
    parser.add_argument('--debug', default=False, type=str2bool, help='debug mode for visualizing ')
    args = parser.parse_args()

    group = args.name[:-7] # remove the seed from the name

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
    
    # Save config
    config_path = os.path.join(args.ckpt_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Check for existing checkpoint to resume
    ckpt_path = get_last_ckpt(args.ckpt_dir)
    
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        wandb_id = ckpt['wandb_id']
        wandb.init(project='test', id=wandb_id, resume='must', group=group)
        main(args, ckpt)
    else:
        print(f"Starting new training run: {args.name}")
        wandb.init(entity=args.wandb_entity, project=args.wandb_project_name, name=args.name, config=vars(args), group=group)
        main(args)
