import os, random, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import einops
import json
from contextlib import nullcontext
import h5py
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as TF
from cam_embedding import PluckerEmbedder
from viz_utils import project_world_point_to_pixel_cam_to_world, _extract_eef_world_pos, draw_clipped_arrow_fixed_head
import cv2
import argparse

def to_mp4(save_path, image_list, reward_list=None, success_list=None, info_list=None):
    """
    Save a list of images as an MP4 video with reward and success overlaid using imageio with H264 encoding.
    """
    import imageio
    import os

    # Convert images to list of numpy arrays if needed
    if isinstance(image_list, torch.Tensor):
        image_list = image_list.cpu().numpy()

    # Ensure the save path has .mp4 extension
    if not save_path.endswith('.mp4'):
        save_path = save_path.replace('.avi', '.mp4')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare frames with overlays
    frames = []
    for i, img in enumerate(image_list):
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Add text overlays if provided
        if reward_list is not None or success_list is not None:
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.load_default()

            y_offset = 10
            if reward_list is not None and i < len(reward_list):
                draw.text((10, y_offset), f"Reward: {reward_list[i]:.3f}", fill=(255, 255, 255), font=font)
                y_offset += 30

            if success_list is not None and i < len(success_list):
                success_text = "SUCCESS" if success_list[i] else "FAILURE"
                color = (0, 255, 0) if success_list[i] else (255, 0, 0)
                draw.text((10, y_offset), success_text, fill=color, font=font)

            img = np.array(img_pil)

        frames.append(img)

    # Save video with H264 encoding, suppress warnings
    import sys

    # Redirect stderr to suppress libx264 warnings
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull

        with imageio.get_writer(save_path, fps=10, codec='h264', ffmpeg_params=['-crf', '23', '-preset', 'medium']) as writer:
            for frame in frames:
                writer.append_data(frame)

        sys.stderr = old_stderr

class Evaluator:
    """
    Class to evaluate policies on robosuite environments
    """
    def __init__(self, env, norm_stats, dataset_path, args):

        self.env = env
        self.args = args
        self.norm_stats = {k: torch.tensor(v).float() for k, v in norm_stats.items()}
        self.chunk_size = args.chunk_size
        self.max_steps = args.eval_max_steps
        self.eval_save_n_video = args.eval_save_n_video

        self.H = 256
        self.W = 256

        self.success_by_seed = {}

        self.plucker_embedder = PluckerEmbedder(img_size=256, device='cuda')
        self.intrinsics = self._get_camera_intrinsics()


        camera_poses_dir = args.camera_poses_dir
        self.num_side_cam = int(args.num_side_cam)
        if not args.default_cam:
            self.camera_poses_by_name = {}
            for filename in args.pose_files:
                poses_path = os.path.join(camera_poses_dir, filename)
                with open(poses_path, 'r') as f:
                    raw = json.load(f)
                pose_name = os.path.splitext(filename)[0]
                self.camera_poses_by_name[pose_name] = raw['poses']
                print(f"Loaded {len(raw['poses'])} camera poses (old format) from {poses_path}; key={pose_name}; num_side_cam={self.num_side_cam}")
        else:
            print("Evaluator: default_cam=True; using agentview pose duplicated if needed")

        # Detect action space from dataset metadata
        with h5py.File(dataset_path, 'r') as f:
            action_space_attr = f['data'].attrs['action_space']
        self.action_space = action_space_attr.decode('utf-8') if isinstance(action_space_attr, bytes) else action_space_attr

    def _get_camera_intrinsics(self):
        """Extract camera intrinsics from robosuite environment."""
        cam_name = "agentview"
        cam_id = self.env.sim.model.camera_name2id(cam_name)

        # Get field of view and image dimensions
        fovy = self.env.sim.model.cam_fovy[cam_id] * np.pi / 180.0
        width, height = 256, 256

        # Compute focal length
        focal_length = height / (2 * np.tan(fovy / 2))

        # Create intrinsics matrix
        intrinsics = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        return intrinsics

    def _get_motion_dynamics_basis(self, cam_to_world: np.ndarray):
        """
        cam_to_world: (4,4) camera pose matrix from pose_set (agentview).
                    This is T_{w<-c} (world_from_camera).

        Returns:
            torch.Tensor (3,2) on CUDA:
                [ [ux, vx],
                [uy, vy],
                [uz, vz] ]
            each row is a unit 2D direction vector in image (u,v) space corresponding to
            +X, +Y, +Z axes of the world/robot frame.
        """

        K = self._get_camera_intrinsics().astype(np.float32)  # (3,3)
        cx, cy = float(K[0, 2]), float(K[1, 2])

        cam_to_world = np.asarray(cam_to_world, dtype=np.float32)
        assert cam_to_world.shape == (4, 4)

        # R_wc: world_from_cam rotation
        R_wc = cam_to_world[:3, :3]          # (3,3)
        # R_cw: cam_from_world rotation
        R_cw = R_wc.T

        # world/robot axes unit directions
        dirs_w = np.stack([
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # +X
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # +Y
            np.array([0.0, 0.0, 1.0], dtype=np.float32),  # +Z
        ], axis=0)  # (3,3)

        eps = 1e-8
        basis_uv = np.zeros((3, 2), dtype=np.float32)

        for i in range(3):
            d_w = dirs_w[i]                       # (3,)
            d_c = (R_cw @ d_w.reshape(3, 1)).reshape(3)  # (3,)

            # homogeneous image direction (vanishing point)
            p = (K @ d_c.reshape(3, 1)).reshape(3)  # (3,)
            denom = float(p[2])
            if abs(denom) < eps:
                denom = eps if denom >= 0 else -eps

            u = float(p[0] / denom)
            v = float(p[1] / denom)

            vec = np.array([u - cx, v - cy], dtype=np.float32)  # direction from principal point
            n = float(np.linalg.norm(vec))
            if n < eps:
                # degenerate fallback
                vec = p[:2].astype(np.float32)
                n = float(np.linalg.norm(vec)) + eps

            basis_uv[i] = vec / (n + eps)

        return torch.from_numpy(basis_uv).float()

    def make_motion_basis_axis_rgb_tensor_cam_to_world(
        self,
        rgb_tensor: torch.Tensor,                  # (3,H,W) in [0,1]
        motion_dynamics_basis: torch.Tensor,        # (3,2) or (6,)
        cam_to_world: np.ndarray | torch.Tensor | None = None,   # (4,4)
        robot_eef_abs_poses: np.ndarray | torch.Tensor | None = None,
        origin_robot: bool = False,               # True면 eef 위치를 origin으로 사용
        origin_fallback: str = "pp",               # "pp" or "center"
        arrow_len: int = 60,
        line_thickness: int = 2,
        return_overlay: bool = False,              # True면 rgb 위에 그려서 반환
        overlay_alpha: float = 0.85,
    ):
        """
        Returns:
        axis_tensor: (3,H,W) float in [0,1] on same device as rgb_tensor
        origin_xy: (ox,oy) int tuple
        """
        H, W = int(rgb_tensor.shape[1]), int(rgb_tensor.shape[2])

        # basis -> (3,2) numpy
        if motion_dynamics_basis.ndim == 1:
            basis = motion_dynamics_basis.view(3, 2)
        else:
            basis = motion_dynamics_basis
        basis_np = basis.detach().float().cpu().numpy()

        # 1) origin from EEF projection if available
        ox = oy = None
        if origin_robot==True and cam_to_world is not None and robot_eef_abs_poses is not None:
            if isinstance(cam_to_world, torch.Tensor):
                c2w = cam_to_world.detach().cpu().numpy()
            else:
                c2w = np.asarray(cam_to_world)

            if c2w.shape != (4, 4):
                raise ValueError(f"cam_to_world must be (4,4), got {c2w.shape}")

            K = self._get_camera_intrinsics()
            p_world = _extract_eef_world_pos(robot_eef_abs_poses)
            uv = project_world_point_to_pixel_cam_to_world(K, c2w, p_world)
            # uv = project_world_point_to_pixel_CU_cam_to_world(K, c2w, p_world)
            if uv is not None:
                u, v = uv
                ox = int(round(u)); oy = int(round(v))
                ox = max(0, min(W - 1, ox))
                oy = max(0, min(H - 1, oy))

        # 2) fallback origin
        if ox is None or oy is None:
            if origin_fallback == "pp":
                K = self._get_camera_intrinsics()
                ox, oy = int(round(float(K[0, 2]))), int(round(float(K[1, 2])))
                ox = max(0, min(W - 1, ox))
                oy = max(0, min(H - 1, oy))
            elif origin_fallback == "center":
                ox, oy = W // 2, H // 2
            else:
                raise ValueError(f"origin_fallback must be 'pp' or 'center'")

        # draw base
        if return_overlay:
            base_rgb = (rgb_tensor[:3].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        else:
            base_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        img_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: x=red y=green z=blue
        labels = ["x", "y", "z"]

        origin_xy = (ox, oy)
        cv2.circle(img_bgr, origin_xy, 3, (255, 255, 255), -1)

        for i in range(3):
            du = float(basis_np[i, 0])
            dv = -float(basis_np[i, 1])  # image v-axis flip

            end_xy = (int(round(ox + arrow_len * du)),
                    int(round(oy + arrow_len * dv)))

            ok, c0, c1 = draw_clipped_arrow_fixed_head(
                img_bgr,
                origin_xy,
                end_xy,
                colors[i],
                thickness=line_thickness,
                head_len_px=8,   # ✅ 여기서 머리 크기 조절 (픽셀)
                head_w_px=6,
            )
            # cv2.arrowedLine(img_bgr, origin_xy, end_xy, colors[i], line_thickness, tipLength=0.01)
            # cv2.putText(img_bgr, labels[i], end_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # uint8

        if return_overlay:
            # optional smoother blending with original rgb
            rgb_u8 = (rgb_tensor[:3].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            out_rgb = (overlay_alpha * out_rgb + (1.0 - overlay_alpha) * rgb_u8).astype(np.uint8)

        axis_tensor = torch.from_numpy(out_rgb).float().permute(2, 0, 1) / 255.0
        axis_tensor = axis_tensor.to(device=rgb_tensor.device)

        return axis_tensor, (ox, oy)

    def _set_camera_pose(self, cam_to_world):
        """Set camera pose in robosuite environment."""

        cam_name = "agentview"
        cam_id = self.env.sim.model.camera_name2id(cam_name)

        # Set camera position
        self.env.sim.model.cam_pos[cam_id] = cam_to_world[:3, 3]

        # Set camera orientation (convert rotation matrix to quaternion)
        rotation = Rotation.from_matrix(cam_to_world[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        # MuJoCo uses [w, x, y, z] format
        self.env.sim.model.cam_quat[cam_id] = [quat[3], quat[0], quat[1], quat[2]]

    def _render_cam_image(self, cam_pose_raw):
        if cam_pose_raw is not None:
            cam_pose = np.array(cam_pose_raw)
            self._set_camera_pose(cam_pose)
        self.env.sim.forward()
        img = self.env.sim.render(camera_name="agentview", height=256, width=256, depth=False)
        return np.flipud(img).copy()

    def _image_to_tensor(self, cam_img, cam_pose_raw):
        rgb_tensor = einops.rearrange(torch.from_numpy(cam_img).float() / 255.0, 'h w c -> c h w')
        if self.args.use_plucker and cam_pose_raw is not None:
            cam_pose = np.array(cam_pose_raw)
            intrinsics_tensor = torch.from_numpy(self.intrinsics).unsqueeze(0).float().cuda()
            cam_to_world_tensor = torch.from_numpy(cam_pose).unsqueeze(0).float().cuda()
            with torch.no_grad():
                plucker_data = self.plucker_embedder(intrinsics_tensor, cam_to_world_tensor)
                plucker_tensor = einops.rearrange(plucker_data['plucker'][0].cpu(), 'h w c -> c h w')

            if self.args.use_dynamics_basis:
                motion_dynamics_basis = self._get_motion_dynamics_basis(cam_pose).reshape(-1)  # (6,)
                sid = self.env.sim.model.site_name2id("gripper0_right_grip_site")
                robot_eef_pos = self.env.sim.data.site_xpos[sid].copy()
                axis_tensor, origin_xy = self.make_motion_basis_axis_rgb_tensor_cam_to_world(
                    rgb_tensor=rgb_tensor,                  # (3,H,W)
                    motion_dynamics_basis=motion_dynamics_basis,
                    cam_to_world=cam_pose,                  # cam_pose = cam_to_world (고정)
                    robot_eef_abs_poses=robot_eef_pos,  # 네가 가진 eef pose
                    origin_robot=self.args.use_basis_origin_robot,
                    origin_fallback="pp",
                    arrow_len=60,
                    return_overlay=self.args.use_overlay_basis,
                )
                if self.args.use_overlay_basis:
                    rgb_tensor = axis_tensor
                else:
                    if self.args.use_plucker:
                        # append as extra channels
                        # 3 (RGB) + 6 (Plucker) + 3 (basis) = 12 channels
                        plucker_tensor = axis_tensor
                    else:
                        assert False, "Either use_plucker or use_dynamics_basis must be True"
                # expand to H,W
                # plucker_tensor = motion_dynamics_basis.unsqueeze(-1).unsqueeze(-1).expand(-1, rgb_tensor.shape[1], rgb_tensor.shape[2])
        else:
            plucker_tensor = torch.zeros(6, rgb_tensor.shape[1], rgb_tensor.shape[2])
        return torch.cat([rgb_tensor, plucker_tensor], dim=0)

    def evaluate(self, policy, save_path, video_prefix, pose_name, init_state=None, episode_num=0):
        np.random.seed(episode_num)

        # Initialize environment
        if init_state is not None:
            self.env.reset()
            self.env.sim.set_state_from_flattened(init_state)
        else:
            self.env.reset()

        if self.action_space in ('eef_delta', 'joint_delta'):
            self.env.set_init_action()

        camera_frames, success_labels, rewards, success = [], [], [], []
        done = False
        step = 0
        has_succeeded = False

        if self.args.default_cam:
            pose_set = [None] * self.num_side_cam
        else:
            poses_list = self.camera_poses_by_name[pose_name]
            if self.num_side_cam == 1:
                pose_set = [poses_list[episode_num]]
            else:
                pose_set = [poses_list[2 * episode_num], poses_list[2 * episode_num + 1]]

        while not done and step < self.max_steps:
            per_cam_images = [self._render_cam_image(p) for p in pose_set]
            camera_frame = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
            camera_frames.append(camera_frame)
            success_labels.append(has_succeeded)

            # Build model input [1, num_cameras, C, H, W]
            per_cam_tensors = [self._image_to_tensor(img, p) for img, p in zip(per_cam_images, pose_set)]
            image_tensor = torch.stack(per_cam_tensors, dim=0).unsqueeze(0).cuda()

            # Camera extrinsics: always two entries, shape [1, 2, 4, 4]
            if self.args.use_cam_pose and not self.args.default_cam:
                per_cam_T = []
                for i in range(2):
                    if i < len(pose_set) and pose_set[i] is not None:
                        per_cam_T.append(torch.from_numpy(np.array(pose_set[i], dtype=np.float32)).float().cuda())
                    else:
                        per_cam_T.append(torch.zeros(4, 4, device='cuda'))
                cam_extrinsics = torch.stack(per_cam_T, dim=0).unsqueeze(0)
            else:
                cam_extrinsics = torch.zeros(1, 2, 4, 4, device='cuda')

            state_vector = self.env.sim.data.qpos[:7]
            if np.random.rand() < self.args.prob_drop_proprio:
                state_vector = np.zeros_like(state_vector)
            normalized_state = (state_vector - self.norm_stats["state_mean"].cpu().numpy()) / self.norm_stats["state_std"].cpu().numpy()
            state_tensor = einops.rearrange(torch.tensor(normalized_state, device="cuda").float(), 'd -> 1 d')

            with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if self.args.use_fp16 else nullcontext()):
                action_chunk = policy({'qpos': state_tensor, 'image': image_tensor, 'cam_extrinsics': cam_extrinsics})
            action_chunk = action_chunk[0].float().cpu().numpy() * self.norm_stats["action_std"].cpu().numpy() + self.norm_stats["action_mean"].cpu().numpy()

            # Execute action chunk
            for i in range(action_chunk.shape[0]):
                if done or step >= self.max_steps:
                    break

                next_obs, reward, done, info = self.env.step(action_chunk[i])

                current_success = (reward == 1)
                has_succeeded = has_succeeded or current_success
                rewards.append(float(reward))
                success.append(current_success)
                step += 1

                if episode_num < self.eval_save_n_video and i < action_chunk.shape[0] - 1:
                    per_cam_images = [self._render_cam_image(p) for p in pose_set]
                    camera_frame = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
                    camera_frames.append(camera_frame)
                    success_labels.append(has_succeeded)

        final_success = any(success)
        print(f"Episode {episode_num}: Success = {final_success}")

        self.success_by_seed[episode_num] = bool(final_success)

        if episode_num < self.eval_save_n_video:
            camera_video_path = os.path.join(save_path, f"{video_prefix}_{pose_name}_success_{final_success}.mp4")
            to_mp4(camera_video_path, camera_frames, success_list=success_labels)

        results = {
            "success_rate": float(final_success),
            "mean_episode_length": float(step),
            "max_rewards": rewards
        }

        return results, float(final_success), step
