# Residual Copilot Deployment Code

Open source hardware deployment codebase for the paper: Efficient and Reliable Teleoperation through Real-to-Sim-to-Real Shared Autonomy.

**[Paper](https://residual-copilot.github.io/files/paper.pdf) | [Project Page](https://residual-copilot.github.io) | [Sim Code](https://github.com/shuosha/Residual_Copilot) | [Data & Checkpoints](https://huggingface.co/collections/shashuo0104/residual-copilot)**

---

## Table of Contents

- [Installation](#installation)
  - [Software](#software)
  - [Hardware](#hardware)
- [Setup](#setup)
- [Running the System](#running-the-system)
- [Data Collection](#data-collection)
- [Codebase Structure](#codebase-structure)

---

## Installation

### Software

#### 1. Python Environment (Python 3.9)

```bash
git clone --recurse-submodule https://github.com/shuosha/Residual_Copilot_Deployment.git
cd Residual_Copilot_Deployment
bash scripts/setup.sh
source .venv/bin/activate
```

<details>
<summary>Manual setup (if you need to run steps individually)</summary>

```bash
git clone --recurse-submodule https://github.com/shuosha/Residual_Copilot_Deployment.git
cd Residual_Copilot_Deployment
uv venv --python 3.9 && source .venv/bin/activate
uv sync
uv pip install --no-deps rl-games==1.6.1 && uv pip install gym==0.23.1
uv pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
uv pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```

`uv sync` installs all dependencies including PyTorch 2.1.0+cu121. The remaining packages have conflicting or missing build deps and are installed separately:
- `rl-games` — pins incompatible torch/gym versions, installed with `--no-deps`
- `nvdiffrast`, `pytorch3d` — require torch at build time, installed with `--no-build-isolation`

</details>

#### 2. Intel RealSense SDK + Viewer

```bash
# Install system dependencies
sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
sudo apt-get install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev v4l-utils
sudo apt-get install git wget cmake build-essential
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at

# Build and install librealsense (unplug cameras first)
cd src/third_party/librealsense
git checkout v2.56.5
./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true \
  -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=$(which python)
cd ../../../
sudo make uninstall && make clean && make -j8 && sudo make install

# Verify
realsense-viewer
```

#### 3. DYNAMIXEL Wizard 2.0 (for GELLO servo configuration)

Download from https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/

```bash
sudo chmod 775 DynamixelWizard2Setup_x64
./DynamixelWizard2Setup_x64
```

Add your user to the `dialout` group for USB serial access:

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

#### 4. UFactory Studio

Install [UFactory Studio](https://www.ufactory.cc/ufactory-studio/) for XArm configuration, firmware updates, and manual jogging.

### Hardware

| Component | Details |
|-----------|---------|
| Robot arm | UFactory XArm7 |
| Wrist camera | Intel RealSense D405 (mounted on end-effector) |
| External camera | Intel RealSense D455 (fixed, viewing the workspace) |
| Teleoperation input | GELLO haptic controller (Dynamixel servos, FTDI USB-serial) |

---

## Setup

### 1. Connect Hardware

- **XArm7**: Connect via Ethernet to the workstation. Default IP: `192.168.1.196`.
- **RealSense cameras**: Connect both D405 (wrist) and D455 (front) via USB 3.0. Verify with `realsense-viewer`.
- **GELLO**: Connect via USB. Verify with `ls /dev/ttyUSB*` (typically `/dev/ttyUSB0`).

### 2. Calibrate GELLO

```bash
python scripts/gello_get_offset.py \
  --start-joints 0 -0.79 0 0.52 0 1.31 0 \
  --joint-signs 1 1 1 1 1 1 1 \
  --port /dev/ttyUSB0
```

### 3. Set Up Foundation Pose Directory

Required for shared autonomy mode (FoundationPose object tracking). Run once per task.

#### Prerequisites

Before running the setup script, update the following hardcoded constants:

1. **Camera serial numbers** — in both `scripts/setup_foundation_pose_dir.py` (`WRIST_SERIAL`, `FRONT_SERIAL`) and `scripts/calibrate.py` (`serials` dict). Find your serials with:

   ```bash
   python -c "import pyrealsense2 as rs; ctx = rs.context(); [print(f'{d.get_info(rs.camera_info.name)}: {d.get_info(rs.camera_info.serial_number)}') for d in ctx.devices]"
   ```

2. **ChArUco board parameters** — in `scripts/calibrate.py`, update the board settings to match your printed calibration board:

   ```python
   ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
   CALIB_BOARD = cv2.aruco.CharucoBoard(
       size=(4, 5),            # (columns, rows) of chessboard squares
       squareLength=0.05,      # square side length in meters
       markerLength=0.036,     # ArUco marker side length in meters
       dictionary=ARUCO_DICT,
   )
   ```

   You can generate a board at https://calib.io/pages/camera-calibration-pattern-generator.

#### Running the setup

```bash
# Full pipeline (first task on new hardware — runs calibration + capture + annotation):
python scripts/setup_foundation_pose_dir.py gearmesh --robot-ip 192.168.1.196

# Subsequent tasks (reuse calibration from first task):
python scripts/setup_foundation_pose_dir.py peginsert \
  --copy-calibration logs/foundation_pose_dir/gearmesh

python scripts/setup_foundation_pose_dir.py nutthread \
  --copy-calibration logs/foundation_pose_dir/gearmesh
```

The setup script walks through these steps:

1. **meshes** — Downloads task-specific `.obj` files from HuggingFace
2. **calibrate** — Hand-eye camera calibration (once per hardware setup; requires both cameras + ChArUco board)
3. **capture** — Captures `empty_scene.png` (no objects) and `detect_roi.jpg` (with objects)
4. **corners** — Interactive annotation of object ROI corners on the detect_roi image

Steps 1, 3, 4 must be repeated for each task (different objects). Step 2 can be shared across tasks if the cameras haven't moved. You can run individual steps with `--steps meshes,capture,corners`.

### 4. Ready

You are now ready to teleoperate, run policy inference, and collect data.

---

## Running the System

### Scripts

All run scripts follow the pattern `python scripts/run_<mode>.py <experiment_name> [options]`.

#### `run_teleop.py` — Teleoperation

Manually control the robot using GELLO or keyboard input.

```bash
# GELLO teleoperation (default)
python scripts/run_teleop.py my_experiment

# Keyboard Cartesian control
python scripts/run_teleop.py my_experiment --input_mode keyboard

# Custom robot IP
python scripts/run_teleop.py my_experiment --robot_ip 192.168.1.196
```

#### `run_shared_autonomy.py` — Shared Autonomy (Residual Copilot)

Human teleop augmented with a learned residual copilot policy. Supports online (GELLO + Copilot) and offline (kNN Pilot + Copilot) modes.

```bash
# Online: human GELLO input + residual copilot correction
python scripts/run_shared_autonomy.py my_experiment \
  --checkpoint_path GearMesh_Residual_Copilot

# Online with residual BC mode available
python scripts/run_shared_autonomy.py my_experiment \
  --checkpoint_path NutThread_Residual_Copilot \
  --res_bc_path NutThread_Residual_BC

# Offline: kNN pilot + residual copilot (no GELLO needed)
python scripts/run_shared_autonomy.py my_experiment \
  --checkpoint_path PegInsert_Residual_Copilot --offline
```

Keyboard controls during shared autonomy:

| Key | Action |
|-----|--------|
| `p` | Pause / unpause teleop |
| `1` | Switch to teleop-only mode |
| `2` | Switch to residual copilot mode |
| `3` | Switch to residual BC mode |
| `m` | Re-track objects (reset FoundationPose) |

Checkpoint names are auto-resolved from HuggingFace. Valid abbreviated names: `GearMesh_Residual_Copilot`, `PegInsert_Residual_Copilot`, `NutThread_Residual_Copilot`, and `*_Residual_BC` variants.

#### `run_policy.py` — Policy Rollout

Run an autonomous policy checkpoint on the robot (no human input).

```bash
python scripts/run_policy.py my_experiment --ckpt /path/to/checkpoint.pth
```

#### `run_replay.py` — Trajectory Replay

Replay recorded trajectories from a `.npz` file on the robot.

```bash
python scripts/run_replay.py my_experiment /path/to/trajectories.npz
```

### Post-processing

After any recording session, synchronize timestamps across camera frames and robot state:

```bash
# Synchronize timestamps
python scripts/postprocess.py logs/<data_dir>/<experiment_name>

# Synchronize + pack into .npy episode files
python scripts/postprocess.py logs/<data_dir>/<experiment_name> --pack
```

---

## Data Collection

### Recording Controls

During teleoperation (`run_teleop.py`) and shared autonomy (`run_shared_autonomy.py`), use these keyboard keys to control recording:

| Key | Action |
|-----|--------|
| `,` (comma) | Start recording an episode |
| `.` (period) | Stop recording — mark episode as **success** |
| `/` (slash) | Stop recording — mark episode as **failed** |
| `p` | Pause / unpause robot control |
| `Esc` | Quit |

Typical workflow:
1. Press `,` to start recording
2. Perform the task
3. Press `.` if the task succeeded, or `/` if it failed
4. Repeat for multiple episodes

### Policy Replay Recording

When running `run_replay.py`, data is recorded automatically for the duration of the replayed trajectory. No manual start/stop is needed.

### Post-processing

After collecting data, always run post-processing to synchronize camera frames with robot state timestamps:

```bash
python scripts/postprocess.py logs/<data_dir>/<experiment_name>
```

Add `--pack` to also pack episode trajectories into `.npy` files for training:

```bash
python scripts/postprocess.py logs/<data_dir>/<experiment_name> --pack
```

---

## Codebase Structure

```
scripts/
  run_teleop.py                  # Teleoperation (GELLO / keyboard)
  run_shared_autonomy.py         # Shared autonomy (human + residual copilot)
  run_policy.py                  # Autonomous policy rollout
  run_replay.py                  # Trajectory replay
  postprocess.py                 # Synchronize timestamps + pack episodes
  calibrate.py                   # Hand-eye camera calibration
  setup_foundation_pose_dir.py   # Full perception setup pipeline
  vis/                           # Visualization utilities

src/robot_control/
  agents/                        # Action agents (run as separate processes)
    action_agent.py              #   Base class: keyboard listener, UDP command sending
    teleop_agent.py              #   GELLO / keyboard teleoperation
    shared_autonomy_agent.py     #   Residual copilot (GELLO + RL residual)
    policy_agent.py              #   Autonomous policy rollout
    gello_listener.py            #   Dynamixel GELLO reader (multiprocess)
    knn_pilot.py                 #   kNN nearest-neighbor action retriever

  env/                           # Environment wrappers (orchestrate agents, cameras, recording)
    xarm_env.py                  #   Base XArm environment (robot connection, camera init)
    robot_env.py                 #   Robot environment (control loop, data storage)
    teleop_env.py                #   Teleoperation environment
    shared_autonomy_env.py       #   Shared autonomy environment (FoundationPose + residual)
    policy_env.py                #   Policy rollout environment
    replay_env.py                #   Trajectory replay environment
    cfg/                         #   Configuration files (env.yaml, knn_pilot_default.json)

  control/                       # Low-level robot control
    xarm_controller.py           #   XArm joint/Cartesian controller (UDP interface)

  perception/                    # Camera and state estimation
    perception.py                #   Multi-camera capture manager
    state_estimator.py           #   FoundationPose object pose estimation
    calibration/                 #   Hand-eye calibration, RealSense manager
    camera/                      #   Camera drivers and shared memory

  utils/                         # Shared utilities
    postprocesser.py             #   Timestamp synchronization + episode packing
    data_storage.py              #   Episode data I/O
    kinematics_utils.py          #   FK/IK via Pinocchio
    annotate_corners.py          #   Interactive ROI corner annotation
    udp_util.py                  #   UDP sender/receiver
    utils.py                     #   Misc helpers (mkdir, get_root, etc.)

  assets/
    xarm7/                       #   Robot URDF and meshes

logs/                            # Output directory for all recorded data
  foundation_pose_dir/           #   Per-task perception setup (meshes, calibration, corners)
    gearmesh/
    peginsert/
    nutthread/
```
