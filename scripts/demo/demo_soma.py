# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001
import argparse
import copy
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import sys
from pathlib import Path

import cv2
import hydra
import functools

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm

# PyTorch ≥2.6 defaults weights_only=True, but our saved files contain numpy arrays.
torch.load = functools.partial(torch.load, weights_only=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# from body_models.soma_hybrid import SomaLayer as SomaLayer_hybrid
from gem.utils.soma_utils.soma_layer import SomaLayer
from gem.utils.geo_transform import (
    apply_T_on_points,
    compute_cam_angvel,
    compute_cam_tvel,
    compute_T_ayfz2ay,
    get_bbx_xys_from_xyxy,
    normalize_T_w2c,
)
from gem.utils.net_utils import detach_to_cpu, to_cuda
from gem.utils.pylogger import Log
from gem.utils.sam3db_extractor import SAM3DBExtractor
from gem.utils.vitpose_extractor import VitPoseExtractor
from gem.utils.video_io_utils import (
    get_video_lwh,
    get_video_reader,
    get_writer,
    merge_videos_grid_2x2,
    merge_videos_horizontal,
    read_video_np,
    save_video,
)
from gem.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)
from gem.utils.vis.o3d_render import Settings, create_meshes, get_ground
from gem.utils.vis.renderer import get_global_cameras_static_v2, get_ground_params_from_points
from gem.utils.cam_utils import estimate_K, get_a_pred_cam
from gem.utils.kp2d_utils import smooth_bbx_xyxy, render_2d_keypoints

CRF = 23  # 17 is lossless, +6 halves output size


class _AttrDict:
    """Minimal Hydra-cfg-compatible namespace used in folder-mode processing.

    Supports attribute access (cfg.video_path) and dict-style .get() so that
    existing functions (run_preprocess, load_data_dict, render_*) work without
    modification.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, _AttrDict(**v))
            else:
                setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _open_cv2_writer(path, width, height, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))


def _parse_args():
    parser = argparse.ArgumentParser()
    # Single-video mode (mutually exclusive with --folder)
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single input video (.mp4)")
    # Folder / dataset mode
    parser.add_argument(
        "--folder", type=str, default=None,
        help=(
            "Dataset root folder.  Expected layout: "
            "folder/<genre>/<music>/video/*.mp4  "
            "Outputs are written as: "
            "folder/<genre>/<music>/{keypoints,bodymodel,merge_video}/<segment>.pkl"
        ),
    )
    parser.add_argument("--output_root", type=str, default="outputs/demo_soma")
    parser.add_argument("-s", "--static_cam", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render_mhr", action="store_true")

    parser.add_argument("--sam3d_ckpt_path", type=str, default=None)
    parser.add_argument("--sam3d_mhr_path", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--exp", type=str, default="gem_soma_regression")
    parser.add_argument("--retarget", action="store_true", help="Retarget SOMA motion to G1 robot")
    parser.add_argument("--force", action="store_true", help="Re-process segments even if outputs already exist")
    return parser.parse_args()


def _build_cfg(args):
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"

    overrides = [
        f"exp={args.exp}",
        f"video_name={video_path.stem}",
        f"video_path={video_path}",
        f"output_root={args.output_root}",
        f"static_cam={str(args.static_cam).lower()}",
        f"verbose={str(args.verbose).lower()}",
        f"render_mhr={str(args.render_mhr).lower()}",
        "use_wandb=false",
        "task=test",
    ]
    if args.ckpt is not None:
        overrides.append(f"ckpt_path={args.ckpt}")
    if args.sam3d_ckpt_path is not None:
        overrides.append(f"sam3d_ckpt_path={args.sam3d_ckpt_path}")
    if args.sam3d_mhr_path is not None:
        overrides.append(f"sam3d_mhr_path={args.sam3d_mhr_path}")

    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        cfg = compose(config_name="demo_soma", overrides=overrides)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def _copy_video_if_needed(cfg):
    src = Path(cfg.video_path)
    dst = Path(cfg.output_dir) / f"{cfg.video_name}.mp4"
    try:
        if not dst.exists() or get_video_lwh(src)[0] != get_video_lwh(dst)[0]:
            reader = get_video_reader(src)
            writer = get_writer(dst, fps=30, crf=CRF)
            for frame in tqdm(reader, total=get_video_lwh(src)[0], desc="Copy Video"):
                writer.write_frame(frame)
            writer.close()
            reader.close()
        cfg.video_path = str(dst)
    except Exception as exc:
        Log.warning(f"[Input] Video copy failed ({exc}). Using original input path.")
        cfg.video_path = str(src)


@torch.no_grad()
def _run_human_detection(cfg, paths, L, W, H):
    """Run YOLOX + ByteTrack human detection on all frames."""
    from gem.utils.yolox_detector import YOLOXDetector, detect_and_track

    Log.info("[Preprocess] Running human detection with YOLOX + ByteTrack...")
    frames = read_video_np(cfg.video_path)

    yolox = YOLOXDetector(device="cuda")
    bbx_xyxy_np, _ = detect_and_track(frames, yolox)

    bbx_xyxy = torch.from_numpy(bbx_xyxy_np).float()  # (L, 4)
    bbx_xyxy = smooth_bbx_xyxy(bbx_xyxy, window=5)
    # Clamp to image bounds
    bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
    bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
    Log.info(f"[Preprocess] Detection done. Saved {L} bboxes to {paths.bbx}")


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Start")
    video_path = cfg.video_path
    paths = cfg.paths

    L, W, H = get_video_lwh(video_path)
    if not Path(paths.bbx).exists():
        _run_human_detection(cfg, paths, L, W, H)

    bbx_xys = torch.load(paths.bbx)["bbx_xys"]

    if cfg.verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        save_video(
            draw_bbx_xyxy_on_image_batch(bbx_xyxy, video), f"{cfg.output_dir}/0_bbx_overlay.mp4"
        )

    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor(device="cuda:0", pose_type="soma")
        frames = read_video_np(video_path)
        vitpose = vitpose_extractor.extract(frames, bbx_xys, img_ds=1.0, path_type="np")
        torch.save(vitpose, paths.vitpose)
    else:
        vitpose = torch.load(paths.vitpose)

    if cfg.verbose:
        draw_pose = vitpose[0] if isinstance(vitpose, tuple) else vitpose
        video = read_video_np(video_path)
        save_video(
            draw_coco17_skeleton_batch(video, draw_pose, 0.5),
            f"{cfg.output_dir}/0_pose_overlay.mp4",
        )

    if cfg.static_cam:
        L, _, _ = get_video_lwh(video_path)
        eye = torch.eye(4).unsqueeze(0).repeat(L, 1, 1).numpy()
        torch.save(eye, paths.slam)
    elif not Path(paths.slam).exists():
        Log.warning("[Preprocess] VO unavailable. Falling back to static camera trajectory.")
        L, _, _ = get_video_lwh(video_path)
        eye = torch.eye(4).unsqueeze(0).repeat(L, 1, 1).numpy()
        torch.save(eye, paths.slam)

    if not Path(paths.vit_features).exists():
        extractor = SAM3DBExtractor(
            checkpoint_path=cfg.get("sam3d_ckpt_path", None),
            mhr_path=cfg.get("sam3d_mhr_path", None),
            device="cuda:0",
        )
        sam3d_results = extractor.extract_video_features(
            video_path,
            bbx_xys,
            render_mhr=cfg.render_mhr,
        )
        length = sam3d_results["transls"].shape[0]
        K_fullimg = estimate_K(W, H).repeat(length, 1, 1)
        pred_cam = get_a_pred_cam(sam3d_results["transls"], bbx_xys, K_fullimg)
        vit_features = {
            "pose_tokens": sam3d_results["pose_tokens"],
            "pred_cam": pred_cam,
        }
        torch.save(vit_features, paths.vit_features)

    Log.info("[Preprocess] Done")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    vitpose = torch.load(paths.vitpose)
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]
    bbx_xys = torch.load(paths.bbx)["bbx_xys"].clone()

    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        t_w2c = torch.zeros(length, 3)
        T_w2c = torch.eye(4).unsqueeze(0).repeat(length, 1, 1)
    else:
        traj = torch.as_tensor(torch.load(paths.slam)).float()
        if traj.ndim != 3 or traj.shape[-2:] != (4, 4):
            raise RuntimeError(f"Unexpected camera trajectory format: {traj.shape}")
        T_w2c = normalize_T_w2c(traj)
        R_w2c = T_w2c[:, :3, :3]
        t_w2c = T_w2c[:, :3, 3]

    return {
        "meta": [{"vid": Path(cfg.video_path).stem}],
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "cam_tvel": compute_cam_tvel(t_w2c),
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
        "f_imgseq": torch.load(paths.vit_features)["pose_tokens"],
        "noisy_pred_cam": torch.load(paths.vit_features).get("pred_cam", None),
        "has_text": torch.tensor([False]),
        "mask": {
            "valid": torch.ones(length).bool(),
            "has_img_mask": torch.ones(length).bool(),
            "has_2d_mask": torch.ones(length).bool(),
            "has_cam_mask": torch.ones(length).bool(),
            "has_audio_mask": torch.zeros(length).bool(),
            "has_music_mask": torch.zeros(length).bool(),
        },
    }


def _get_body_params(pred, key):
    if key in pred:
        return pred[key]
    if key.startswith("body_params_"):
        suffix = key.replace("body_params_", "")
        target_tail = f"_params_{suffix}"
        for k in pred:
            if k.endswith(target_tail):
                return pred[k]
    raise KeyError(f"Missing `{key}` (or any `*{target_tail}` variant) in prediction output.")


def render_incam(cfg, fps=30):
    import open3d as o3d

    pred = torch.load(cfg.paths.hpe_results)
    body_params_incam = _get_body_params(pred, "body_params_incam")

    device = "cuda:0"
    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device=device,
        identity_model_type="mhr",
        mode="warp",
    )
    with torch.no_grad():
        pred_c_verts = soma(**to_cuda(body_params_incam))["vertices"]
    faces = soma.faces.long().cuda()

    video_path = cfg.video_path
    _, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]  # (3, 3)

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]
    color = torch.tensor([0.4, 0.8, 0.4], device=device)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
    # Light from above-front in OpenCV convention (Y-down, +Z forward)
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.SOFT_SHADOWS, np.array([0.0, 0.7, 0.7])
    )
    # Set camera intrinsics from K
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.01, 100.0, float(width), float(height)
    )
    # OpenCV camera convention: camera at origin, looking +Z, Y-down
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    renderer.scene.camera.look_at(target, eye, up)

    reader = get_video_reader(video_path)
    writer = _open_cv2_writer(cfg.paths.incam_video, width, height, fps)
    for i, img_raw in tqdm(enumerate(reader), total=pred_c_verts.shape[0], desc="Render Incam"):
        if i >= pred_c_verts.shape[0]:
            break
        mesh = create_meshes(pred_c_verts[i], faces, color)
        mesh_name = f"mesh_{i}"
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(mesh_name, mesh, lit_mat)
        rendered = np.array(renderer.render_to_image())  # (H, W, 3) uint8
        depth = np.asarray(renderer.render_to_depth_image())  # (H, W) float32
        mask = (depth < 1.0).astype(np.float32)  # 1.0 where geometry exists
        mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.0)  # soften edges
        alpha = mask[..., np.newaxis]  # (H, W, 1)
        composite = rendered.astype(np.float32) * alpha + img_raw.astype(np.float32) * (1.0 - alpha)
        composite = composite.clip(0, 255).astype(np.uint8)
        writer.write(composite[..., ::-1])
    writer.release()
    reader.close()


def render_global_o3d(cfg, fps=30):
    import open3d as o3d

    pred = torch.load(cfg.paths.hpe_results)
    body_params_global = _get_body_params(pred, "body_params_global")

    device = "cuda:0"
    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device=device,
        identity_model_type="mhr",
        mode="warp",
    )
    with torch.no_grad():
        soma_out_glob = soma(**to_cuda(body_params_global))
    verts_glob = soma_out_glob["vertices"]
    joints_glob = soma_out_glob["joints"]
    faces_soma = soma.faces.long().cuda()

    def move_to_start_point_face_z(verts, joints):
        verts = verts.clone()
        joints = joints.clone()
        y_min = verts[:, :, 1].min()
        verts[:, :, 1] -= y_min
        joints[:, :, 1] -= y_min
        T_ay2ayfz = compute_T_ayfz2ay(joints[[0]], inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        joints = apply_T_on_points(joints, T_ay2ayfz)
        return verts, joints

    verts_glob, joints_glob = move_to_start_point_face_z(verts_glob, joints_glob)
    _, width, height = get_video_lwh(cfg.video_path)
    from gem.utils.cam_utils import create_camera_sensor

    _, _, K = create_camera_sensor(width, height, 50)

    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    ground = get_ground(max(scale, 3) * 1.5, cx, cz)
    position, target, up = get_global_cameras_static_v2(
        verts_glob.cpu().clone(), beta=4.5, cam_height_degree=30, target_center_height=1.0
    )

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]
    color = torch.tensor([0.4, 0.8, 0.4], device=verts_glob.device)
    writer = _open_cv2_writer(cfg.paths.global_video, width, height, fps)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.NO_SHADOWS, np.array([0.577, -0.577, -0.577])
    )
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.1, 100.0, float(width), float(height)
    )
    renderer.scene.camera.look_at(target.cpu().numpy(), position.cpu().numpy(), up.cpu().numpy())

    gv, gf, gc = ground
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    ground_mat = o3d.visualization.rendering.MaterialRecord()
    ground_mat.shader = Settings.LIT
    renderer.scene.add_geometry("mesh_ground", ground_mesh, ground_mat)

    for t in tqdm(range(verts_glob.shape[0]), desc="Render Global"):
        mesh = create_meshes(verts_glob[t], faces_soma, color)
        if t > 0:
            renderer.scene.remove_geometry(f"mesh_{t - 1}")
        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat)
        writer.write(np.array(renderer.render_to_image())[..., ::-1])
    writer.release()


def resolve_ckpt_path(cfg):
    if cfg.ckpt_path is not None and Path(cfg.ckpt_path).exists():
        return cfg.ckpt_path
    from gem.utils.hf_utils import download_checkpoint

    Log.info("[Checkpoint] Not found locally. Downloading from HuggingFace...")
    return download_checkpoint()


def _make_video_cfg(video_path, preprocess_dir, keypoints_path, bodymodel_path,
                    merge_video_path, args):
    """Build a minimal _AttrDict cfg for a single video in folder mode.

    Parameters
    ----------
    video_path      : Path to the source .mp4
    preprocess_dir  : Directory for intermediate files (bbx, vitpose, …)
    keypoints_path  : Final destination for the keypoints .pkl
    bodymodel_path  : Final destination for the body-model .pkl (hpe_results)
    merge_video_path: Final destination for the merged side-by-side .mp4
    args            : Parsed CLI args (for static_cam, verbose, …)
    """
    stem = Path(video_path).stem
    tmp_dir = Path(preprocess_dir) / stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    paths = _AttrDict(
        bbx=str(tmp_dir / "bbx.pt"),
        vitpose=str(tmp_dir / "vitpose.pt"),
        slam=str(tmp_dir / "camera.pt"),
        vit_features=str(tmp_dir / "vit_features.pt"),
        hpe_results=str(bodymodel_path),
        incam_video=str(tmp_dir / f"{stem}_incam.mp4"),
        global_video=str(tmp_dir / f"{stem}_global.mp4"),
        incam_global_horiz_video=str(merge_video_path),
        keypoints=str(keypoints_path),
    )

    return _AttrDict(
        video_path=str(video_path),
        video_name=stem,
        output_dir=str(tmp_dir),
        static_cam=args.static_cam,
        verbose=args.verbose,
        render_mhr=getattr(args, "render_mhr", False),
        sam3d_ckpt_path=getattr(args, "sam3d_ckpt_path", None),
        sam3d_mhr_path=getattr(args, "sam3d_mhr_path", None),
        paths=paths,
    )


@torch.no_grad()
def _process_single_video(video_cfg, model):
    """Run the full GEM-X pipeline on one video using a pre-loaded model.

    Saves:
      - keypoints .pkl  → vitpose + bbx combined dict
      - bodymodel .pkl  → full SOMA prediction (body_params_global/incam + K)
      - merge video     → in-camera and global side-by-side
    """
    run_preprocess(video_cfg)

    fps = int(cv2.VideoCapture(video_cfg.video_path).get(cv2.CAP_PROP_FPS) + 0.5) or 30

    # Optional 2D overlay (only when --verbose)
    if video_cfg.verbose:
        render_2d_keypoints(
            video_path=video_cfg.video_path,
            vitpose_path=video_cfg.paths.vitpose,
            bbx_path=video_cfg.paths.bbx,
            output_path=str(Path(video_cfg.output_dir) / "0_kp2d77_overlay.mp4"),
            fps=fps,
        )

    data = load_data_dict(video_cfg)

    # --- Keypoints pkl ---
    keypoints_path = Path(video_cfg.paths.keypoints)
    if not keypoints_path.exists():
        kp_data = {
            **torch.load(video_cfg.paths.bbx),          # bbx_xyxy, bbx_xys
            "vitpose": torch.load(video_cfg.paths.vitpose),
        }
        torch.save(kp_data, str(keypoints_path))
        Log.info(f"[Output] Keypoints → {keypoints_path}")

    # --- Body model pkl (hpe_results) ---
    bodymodel_path = Path(video_cfg.paths.hpe_results)
    if not bodymodel_path.exists():
        pred = model.predict(data, static_cam=video_cfg.static_cam, postproc=True)
        torch.save(detach_to_cpu(pred), str(bodymodel_path))
        Log.info(f"[Output] Body model → {bodymodel_path}")

    # --- Render videos ---
    render_incam(video_cfg, fps=fps)
    render_global_o3d(video_cfg, fps=fps)

    # --- Merge video ---
    merge_video_path = Path(video_cfg.paths.incam_global_horiz_video)
    if not merge_video_path.exists():
        merge_videos_horizontal(
            [video_cfg.paths.incam_video, video_cfg.paths.global_video],
            str(merge_video_path),
        )
        Log.info(f"[Output] Merge video → {merge_video_path}")


def run_dataset_folder(folder, args):
    """Process an entire dataset folder with the GEM-X pipeline.

    Expected dataset layout (produced by main.py / downloader)::

        <folder>/
          <genre>/
            <music>/
              video/
                seg000.mp4
                seg001.mp4
                ...

    Outputs written into the same ``<music>/`` directory::

        <music>/
          keypoints/          # vitpose + bbx per segment
            seg000.pt
          smpl/               # SOMA body params per segment
            seg000.pt
          video_merge/        # side-by-side rendered video per segment
            seg000.mp4
          preprocess/         # intermediate files (kept for resuming)
            seg000/
              bbx.pt
              vitpose.pt
              camera.pt
              vit_features.pt

    Already-processed segments are skipped automatically.
    """
    folder = Path(folder)
    if not folder.exists():
        Log.error(f"[Folder] Dataset folder not found: {folder}")
        return

    # Discover all video segments in the dataset
    video_files = sorted(folder.glob("*/*/video/*.mp4"))
    if not video_files:
        Log.warning(f"[Folder] No .mp4 files found under {folder}/*/video/")
        return

    Log.info(f"[Folder] Found {len(video_files)} video(s) across the dataset.")

    # --- Load model ONCE using Hydra via the first video ---
    args_first = copy.copy(args)
    args_first.video = str(video_files[0])
    cfg = _build_cfg(args_first)
    ckpt_path = resolve_ckpt_path(cfg)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(ckpt_path)
    model = model.eval().cuda()
    Log.info("[Folder] Model loaded. Starting per-video processing...")

    ok, failed = [], []

    for video_path in video_files:
        # Derive output paths from dataset layout
        music_dir = video_path.parent.parent      # <folder>/<genre>/<music>/
        genre_name = music_dir.parent.name
        music_name = music_dir.name
        stem = video_path.stem

        preprocess_dir = music_dir / "preprocess"
        keypoints_dir  = music_dir / "keypoints"
        bodymodel_dir  = music_dir / "smpl"
        merge_dir      = music_dir / "video_merge"

        keypoints_path   = keypoints_dir  / f"{stem}.pt"
        bodymodel_path   = bodymodel_dir  / f"{stem}.pt"
        merge_video_path = merge_dir      / f"{stem}.mp4"

        # Create output directories
        for d in [preprocess_dir / stem, keypoints_dir, bodymodel_dir, merge_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Skip if all three final outputs already exist (unless --force)
        if not args.force and bodymodel_path.exists() and keypoints_path.exists() and merge_video_path.exists():
            Log.info(f"[Skip] {genre_name}/{music_name}/{stem} — already processed.")
            ok.append(stem)
            continue

        if args.force:
            for p in [keypoints_path, bodymodel_path, merge_video_path]:
                p.unlink(missing_ok=True)
            tmp_dir = preprocess_dir / stem
            for render_file in [tmp_dir / f"{stem}_incam.mp4", tmp_dir / f"{stem}_global.mp4"]:
                render_file.unlink(missing_ok=True)

        Log.info(f"\n{'─' * 60}")
        Log.info(f"[Processing] {genre_name}/{music_name}/{stem}")

        video_cfg = _make_video_cfg(
            video_path=video_path,
            preprocess_dir=preprocess_dir,
            keypoints_path=keypoints_path,
            bodymodel_path=bodymodel_path,
            merge_video_path=merge_video_path,
            args=args,
        )

        try:
            _process_single_video(video_cfg, model)
            ok.append(stem)
        except Exception as exc:
            Log.error(f"[Error] {genre_name}/{music_name}/{stem}: {exc}")
            failed.append((genre_name, music_name, stem))

    Log.info(f"\n{'═' * 60}")
    Log.info(f"[Folder] Done — {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        Log.info("Failed segments:")
        for g, m, s in failed:
            Log.info(f"  • {g}/{m}/{s}")


def main():
    args = _parse_args()

    # ── Folder / dataset mode ──────────────────────────────────────────────
    if args.folder:
        run_dataset_folder(args.folder, args)
        return

    # ── Single-video mode ──────────────────────────────────────────────────
    if not args.video:
        print("[ERROR] Provide either --video <path> or --folder <dataset_root>.")
        import sys; sys.exit(1)

    cfg = _build_cfg(args)
    _copy_video_if_needed(cfg)
    run_preprocess(cfg)
    fps = int(cv2.VideoCapture(cfg.video_path).get(cv2.CAP_PROP_FPS) + 0.5) or 30
    render_2d_keypoints(
        video_path=cfg.video_path,
        vitpose_path=cfg.paths.vitpose,
        bbx_path=cfg.paths.bbx,
        output_path=str(Path(cfg.output_dir) / "0_kp2d77_overlay.mp4"),
        fps=fps,
    )
    data = load_data_dict(cfg)

    if not Path(cfg.paths.hpe_results).exists():
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        ckpt_path = resolve_ckpt_path(cfg)
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()
        pred = model.predict(data, static_cam=cfg.static_cam, postproc=True)
        torch.save(detach_to_cpu(pred), cfg.paths.hpe_results)

    render_incam(cfg, fps=fps)
    render_global_o3d(cfg, fps=fps)

    if args.retarget:
        from scripts.demo.retarget_utils import render_g1_robot, run_retarget

        pred = torch.load(cfg.paths.hpe_results)
        body_params_global = _get_body_params(pred, "body_params_global")
        csv_buffer = run_retarget(body_params_global, fps, cfg.paths.retarget_csv)
        render_g1_robot(cfg, csv_buffer, fps=fps)
        kp2d_video = str(Path(cfg.output_dir) / "0_kp2d77_overlay.mp4")
        merge_videos_grid_2x2(
            [kp2d_video, cfg.paths.incam_video, cfg.paths.global_video, cfg.paths.retarget_video],
            cfg.paths.incam_global_horiz_video,
        )
    else:
        merge_videos_horizontal(
            [cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video
        )
    Log.info(f"[Done] Outputs in {cfg.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    main()
