from pathlib import Path
import os
import time
import numpy as np
import cv2
import open3d as o3d
from threadpoolctl import threadpool_limits
import multiprocess as mp
import threading
from threading import Lock
from typing import Union

from robot_control.utils.utils import get_root
root: Path = get_root(__file__)

from robot_control.perception.camera.multi_realsense import MultiRealsense
from robot_control.perception.camera.single_realsense import SingleRealsense


class Perception(mp.Process):
    name = "Perception"

    def __init__(
        self,
        realsense: Union[MultiRealsense, SingleRealsense], 
        capture_fps, 
        record_fps, 
        record_time, 
        process_func,
        exp_name=None,
        data_dir="data",
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose

        self.capture_fps = capture_fps
        self.record_fps = record_fps
        self.record_time = record_time
        self.exp_name = exp_name
        self.data_dir = data_dir

        if self.exp_name is None:
            assert self.record_fps == 0

        self.realsense = realsense
        self.perception_q = mp.Queue(maxsize=1)

        self.process_func = process_func
        self.num_cam = len(realsense.cameras.keys())
        self.alive = mp.Value('b', False)
        self.record_restart = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)
        self.record_failed = mp.Value('b', False)

        self.record_mode = mp.Value('i', 0)

    def log(self, msg):
        if self.verbose:
            print(f"\033[92m{self.name}: {msg}\033[0m")

    @property
    def can_record(self):
        return self.record_fps != 0

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        realsense = self.realsense

        # i = self.index
        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time

        cameras_output = None
        recording_frame = float("inf")  # local record step index (since current record start), record fps
        record_start_frame = 0  # global step index (since process start), capture fps
        is_recording = False  # recording state flag
        timestamps_f = None

        eps_idx = 0

        while self.alive.value:
            try: 
                cameras_output = realsense.get(out=cameras_output)
                get_time = time.time()
                timestamps = [cameras_output[i]['timestamp'].item() for i in range(self.num_cam)]
                if is_recording and not all([abs(timestamps[i] - timestamps[i+1]) < 0.05 for i in range(self.num_cam - 1)]):
                    print(f"Captured at different timestamps: {[f'{x:.2f}' for x in timestamps]}")

                # treat captured time and record time as the same
                process_start_time = get_time
                process_out = self.process_func(cameras_output) if self.process_func is not None else cameras_output
                self.log(f"process time: {time.time() - process_start_time}")

                # if not self.perception_q.full():
                self.perception_q.put(process_out)

                if self.can_record:
                    # recording state machine:
                    #           ---restart_cmd--->
                    # record                           not_record
                    #       <-- stop_cmd / timeover ----
                    if not is_recording and self.record_restart.value == True:
                        self.record_restart.value = False
                        self.record_failed.value = False

                        recording_frame = 0
                        record_start_time = get_time
                        record_start_frame = cameras_output[0]['step_idx'].item()
                        
                        record_dir = root / "logs" / self.data_dir / self.exp_name / f"{record_start_time:.0f}"
                        os.makedirs(record_dir, exist_ok=True)
                        timestamps_f = open(f'{record_dir}/timestamps.txt', 'a')
                        
                        for i in range(self.num_cam):
                            os.makedirs(f'{record_dir}/camera_{i}/rgb', exist_ok=True)
                            os.makedirs(f'{record_dir}/camera_{i}/depth', exist_ok=True)
                        is_recording = True
                    
                    elif is_recording and (
                        self.record_stop.value == True or 
                        (recording_frame >= record_time * record_fps)
                    ):
                        finish_time = get_time
                        print(f"-------------- eps_idx {eps_idx} --------------")
                        print(f"is_recording {is_recording}, self.record_stop.value {self.record_stop.value}, recording time {recording_frame}, max recording time {record_time} * {record_fps}")
                        print(f"total time: {finish_time - record_start_time}")
                        print(f"fps: {recording_frame / (finish_time - record_start_time)}")
                        eps_idx += 1
                        if self.record_failed.value:
                            np.savetxt(record_dir / "failed.txt", np.array([1]), fmt="%d")
                        if self.record_mode.value != 0:
                            if self.record_mode.value == 1:
                                mode_str = "teleop"
                            elif self.record_mode.value == 2:
                                mode_str = "residual_copilot"
                            elif self.record_mode.value == 3:
                                mode_str = "residual_bc"
                            np.savetxt(record_dir / f"{mode_str}.txt", np.array([1]), fmt="%s")

                        is_recording = False
                    
                        timestamps_f.close()
                        self.record_restart.value = False
                        self.record_stop.value = False
                        self.record_failed.value = False
                    else:
                        self.record_restart.value = False
                        self.record_stop.value = False

                    # record the frame according to the record_fps
                    if is_recording and cameras_output[0]['step_idx'].item() >= (recording_frame * (capture_fps // record_fps) + record_start_frame):
                        timestamps_f.write(' '.join(
                            [str(cameras_output[i]['step_idx'].item()) for i in range(self.num_cam)] + 
                            [str(np.round(cameras_output[i]['timestamp'].item() - record_start_time, 3)) for i in range(self.num_cam)] + 
                            [str(cameras_output[i]['timestamp'].item()) for i in range(self.num_cam)]
                        ) + '\n')
                        timestamps_f.flush()
                        for i in range(self.num_cam):
                            cv2.imwrite(f'{record_dir}/camera_{i}/rgb/{recording_frame:06}.jpg', cameras_output[i]['color'])
                            # NOTE: storing depth happens here
                            # cv2.imwrite(f'{record_dir}/camera_{i}/depth/{recoprding_frame:06}.png', cameras_output[i]['depth'])
                        recording_frame += 1

            # except:
            #     print(f"Perception error")
            #     break
            except BaseException as e:
                print("Perception error: ", e.with_traceback())
                break
                # print(f"\033[91mPerception error\033[0m: {e.with_traceback()}")

        if self.can_record:
            if timestamps_f is not None and not timestamps_f.closed:
                timestamps_f.close()
            finish_time = time.time()
            # print(f"total time: {finish_time - record_start_time}")
            # print(f"fps: {recording_frame / (finish_time - record_start_time)}")
        self.stop()
        print("Perception process stopped")


    def start(self):
        self.alive.value = True
        super().start()

    def stop(self):
        self.alive.value = False
        self.perception_q.close()
    
    def set_record_start(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_restart.value == False
        else:
            self.record_restart.value = True
            # print("record restart cmd received")

    def set_record_stop(self, mode=None):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_stop.value == False
        else:
            self.record_stop.value = True
            if mode is not None:
                if mode == "teleop":
                    self.record_mode.value = 1
                elif mode == "residual_copilot":
                    self.record_mode.value = 2
                elif mode == "residual_bc":
                    self.record_mode.value = 3
                else:
                    raise ValueError(f"Unknown record mode: {mode}")

    def set_record_failed(self, mode=None):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_failed.value == False
        else:
            self.record_failed.value = True
            if mode is not None:
                if mode == "teleop":
                    self.record_mode.value = 1
                elif mode == "residual_copilot":
                    self.record_mode.value = 2
                elif mode == "residual_bc":
                    self.record_mode.value = 3
                else:
                    raise ValueError(f"Unknown record mode: {mode}")
            print("record failed cmd received")


def perception_func_test():

    def process_perception_out(perception):
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # vis.add_geometry(coordinate)

        # Set camera view
        # ctr = vis.get_view_control()
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_front([0, -1, 0])
        # ctr.set_up([0, 0, -1])

        while perception.alive.value:
            print(f"[Vis thread] perception out {time.time()}")
            if not perception.perception_q.empty():
                output = perception.perception_q.get()

                # Visualize the merged point cloud
                # vis.clear_geometries()
                # vis.add_geometry(output["object_pcd"])
                # vis.poll_events()
                # vis.update_renderer()
            time.sleep(1)

    # Create an instance of MultiRealsense or SingleRealsense
    realsense = MultiRealsense(
        resolution=(848, 480),
        capture_fps=30,
        enable_color=True,
        enable_depth=True,
        verbose=False
    )
    realsense.start(wait=True)
    realsense.restart_put(start_time=time.time() + 2)
    time.sleep(10)

    # Create an instance of the Perception class
    perception = Perception(
        realsense=realsense, 
        capture_fps=30, # should be the same as the camera fps
        record_fps=30, # 0 for no record
        record_time=10, # in seconds
        process_func=None,
        verbose=False
    )

    # Start the perception process
    perception.start()

    # Start the echo thread
    vis_thread = threading.Thread(name="vis",target=process_perception_out, args=(perception,))
    vis_thread.start()

    # Start the record
    perception.set_record_start()

    time.sleep(12)

    # Stop the record
    perception.set_record_stop()

    time.sleep(2)

    # Stop the perception process
    perception.stop()
    vis_thread.join()
    realsense.stop()

    # exit
    print("Test finished")

if __name__ == "__main__":
    perception_func_test()
