"""Set up a foundation_pose_dir for a new task on new hardware.

Pipeline steps:
  1. meshes     — Download .obj meshes from residual_copilot_assets (per task)
  2. calibrate  — Hand-eye camera calibration → extrinsics.json + intrinsics.json
                  (only needs to be done once per hardware setup; reuse with --copy-calibration)
  3. capture    — Capture two images from the front camera (per task):
                    empty_scene.png  — workspace with NO objects (background reference)
                    detect_roi.jpg   — workspace WITH objects placed for the task
  4. corners    — Annotate ROI corners on detect_roi.jpg → corners.json (per task)
                  Click 4 corners around each object to define the detection region.

Steps 1, 3, 4 must be repeated for each task (different objects).
Step 2 can be shared across tasks if the cameras haven't moved.

Before running, update the hardware-specific constants:
  1. Camera serial numbers (WRIST_SERIAL, FRONT_SERIAL below):
       python -c "import pyrealsense2 as rs; ctx = rs.context(); [print(f'{d.get_info(rs.camera_info.name)}: {d.get_info(rs.camera_info.serial_number)}') for d in ctx.devices]"
  2. ChArUco board parameters in scripts/calibrate.py (size, squareLength, markerLength, dictionary)
     to match your printed calibration board.

Usage:
    # Full pipeline (requires robot + both cameras):
    python scripts/setup_foundation_pose_dir.py gearmesh --robot-ip 192.168.1.196

    # Skip calibration (reuse existing extrinsics/intrinsics from another task):
    python scripts/setup_foundation_pose_dir.py peginsert --copy-calibration logs/foundation_pose_dir/gearmesh

    # Run individual steps:
    python scripts/setup_foundation_pose_dir.py nutthread --steps meshes,capture,corners
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2

# =====================================================================
# Camera serial numbers — update these for your hardware setup.
# Find serials by running:
#   python -c "import pyrealsense2 as rs; ctx = rs.context(); [print(f'{d.get_info(rs.camera_info.name)}: {d.get_info(rs.camera_info.serial_number)}') for d in ctx.devices]"
# =====================================================================
WRIST_SERIAL = "130322273767"   # Intel RealSense D405 (wrist-mounted)
FRONT_SERIAL = "239222300843"   # Intel RealSense D455 (front-facing)


# Task → (held_obj_filename, fixed_obj_filename) on HuggingFace
# These are the .obj mesh files in residual_copilot_assets/objects/
TASK_MESH_FILES = {
    "gearmesh": ("objects/gear.obj", "objects/gear_base.obj"),
    "peginsert": ("objects/peg.obj", "objects/peg_base.obj"),
    "nutthread": ("objects/nut.obj", "objects/nut_base.obj"),
}

HF_ASSETS_REPO = "shashuo0104/residual_copilot_assets"

ALL_STEPS = ["meshes", "calibrate", "capture", "corners"]


def download_meshes(task: str, out_dir: Path):
    """Download held and fixed .obj meshes from HuggingFace."""
    from huggingface_hub import hf_hub_download

    held_hf, fixed_hf = TASK_MESH_FILES[task]
    held_name = os.path.basename(held_hf)
    fixed_name = os.path.basename(fixed_hf)

    print(f"\n[meshes] Downloading {held_hf} and {fixed_hf} from {HF_ASSETS_REPO}...")
    for hf_path, local_name in [(held_hf, held_name), (fixed_hf, fixed_name)]:
        local = hf_hub_download(repo_id=HF_ASSETS_REPO, filename=hf_path, repo_type="dataset")
        dst = out_dir / local_name
        shutil.copy2(local, dst)
        print(f"  -> {dst}")

    print("[meshes] Done.")


def run_calibration(out_dir: Path, robot_ip: str, wrist_serial: str, front_serial: str, show_gui: bool):
    """Run full camera calibration (fixed + handeye + compose)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from calibrate import HandEyeCalibrator

    print(f"\n[calibrate] Running full calibration (robot_ip={robot_ip})...")
    print(f"  wrist serial: {wrist_serial}")
    print(f"  front serial: {front_serial}")
    print("  Ensure both RealSense cameras are connected and the ChArUco board is visible.")

    serials = {"wrist": wrist_serial, "front": front_serial}
    cal = HandEyeCalibrator(work_dir=out_dir, robot_ip=robot_ip, show_gui=show_gui, init_robot=True)
    try:
        cal.fixed_calibration(serials)
        cal.capture_wrist_images(serials.get("wrist"))
        cal.handeye_calibration()
        cal.compose()
    finally:
        cal.close()

    print("[calibrate] Done. Wrote extrinsics.json and intrinsics.json.")


def copy_calibration(out_dir: Path, src_dir: Path):
    """Copy extrinsics.json and intrinsics.json from an existing foundation_pose_dir."""
    print(f"\n[calibrate] Copying calibration from {src_dir}...")
    for fname in ("extrinsics.json", "intrinsics.json"):
        src = src_dir / fname
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping.")
            continue
        dst = out_dir / fname
        shutil.copy2(src, dst)
        print(f"  -> {dst}")
    print("[calibrate] Done.")


def capture_scene_images(out_dir: Path, front_serial: str):
    """Capture empty_scene.png and detect_roi.jpg from the front camera."""
    import pyrealsense2 as rs
    from robot_control.perception.calibration.realsense_manager import RealSenseManager

    print(f"\n[capture] Connecting to front camera (serial={front_serial})...")
    cam = RealSenseManager(
        serial=front_serial,
        width=848, height=480, fps=30,
        depth_format=rs.format.z16,
        color_format=rs.format.bgr8,
    )

    try:
        # Warm up
        for _ in range(30):
            cam.poll_frames()

        # Empty scene — background reference with NO objects
        input("\n[capture] Clear the workspace — remove ALL objects, then press Enter...")
        cam.poll_frames()
        bgr = cam.get_color_image()
        empty_path = out_dir / "empty_scene.png"
        cv2.imwrite(str(empty_path), bgr)
        print(f"  Saved empty_scene.png (background, no objects)")

        # Detect ROI scene — WITH task objects placed on the workspace
        input("\n[capture] Place the task objects on the workspace, then press Enter...")
        cam.poll_frames()
        bgr = cam.get_color_image()
        roi_path = out_dir / "detect_roi.jpg"
        cv2.imwrite(str(roi_path), bgr)
        print(f"  Saved detect_roi.jpg (scene with objects for ROI annotation)")

    finally:
        cam.stop()

    print("[capture] Done.")


def annotate_corners(out_dir: Path):
    """Interactive corner annotation for held and fixed assets. Must be done per task."""
    from robot_control.utils.annotate_corners import identify_ROIs

    roi_path = out_dir / "detect_roi.jpg"
    if not roi_path.exists():
        print(f"\n[corners] ERROR: {roi_path} not found. Run the 'capture' step first.")
        return

    print(f"\n[corners] Opening {roi_path} for ROI annotation...")
    print("  This must be done for each task (different objects = different corners).")
    print("  Click 4 corners around each object to define the detection region.")
    print("  Order: fixed_asset (base/bolt/hole) first, then held_asset (gear/peg/nut).")
    corners = identify_ROIs(
        path=str(roi_path),
        obj_list=["fixed_asset", "held_asset"],
    )

    corners_path = out_dir / "corners.json"
    with open(corners_path, "w") as f:
        json.dump(corners, f, indent=2)
    print(f"  Saved {corners_path}")
    print("[corners] Done.")


def verify_directory(out_dir: Path, task: str):
    """Check that all required files are present."""
    held_name, fixed_name = [os.path.basename(f) for f in TASK_MESH_FILES[task]]
    required = [
        "extrinsics.json",
        "intrinsics.json",
        "empty_scene.png",
        "detect_roi.jpg",
        "corners.json",
        held_name,
        fixed_name,
    ]

    print(f"\n[verify] Checking {out_dir}...")
    all_ok = True
    for fname in required:
        path = out_dir / fname
        status = "OK" if path.exists() else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"  [{status}] {fname}")

    if all_ok:
        print("\nAll files present. foundation_pose_dir is ready.")
    else:
        print("\nSome files are missing. Re-run the corresponding steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Set up a foundation_pose_dir for a new task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task", choices=list(TASK_MESH_FILES.keys()),
                        help="Task name (gearmesh, peginsert, nutthread)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: logs/foundation_pose_dir/<task>)")
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help=f"Comma-separated steps to run (default: {','.join(ALL_STEPS)})")
    parser.add_argument("--copy-calibration", default=None, type=str,
                        help="Copy calibration from existing foundation_pose_dir instead of running calibration")

    # Camera + robot args
    parser.add_argument("--robot-ip", default="192.168.1.196", help="xArm robot IP")
    parser.add_argument("--show-gui", action="store_true", help="Show GUI during calibration")

    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",")]
    for s in steps:
        if s not in ALL_STEPS:
            parser.error(f"Unknown step '{s}'. Valid steps: {ALL_STEPS}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("logs/foundation_pose_dir") / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Step 1: Download meshes
    if "meshes" in steps:
        download_meshes(args.task, out_dir)

    # Step 2: Calibration
    if "calibrate" in steps:
        if args.copy_calibration:
            copy_calibration(out_dir, Path(args.copy_calibration))
        else:
            run_calibration(out_dir, args.robot_ip, WRIST_SERIAL, FRONT_SERIAL, args.show_gui)

    # Step 3: Capture empty scene + detect ROI
    if "capture" in steps:
        capture_scene_images(out_dir, FRONT_SERIAL)

    # Step 4: Annotate corners
    if "corners" in steps:
        annotate_corners(out_dir)

    # Final verification
    verify_directory(out_dir, args.task)


if __name__ == "__main__":
    main()
