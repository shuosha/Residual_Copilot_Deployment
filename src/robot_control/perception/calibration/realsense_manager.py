import time

import numpy as np
import pyrealsense2 as rs


class RealSenseManager:
    def __init__(
        self, serial: str, width: int, height: int, fps: int, depth_format=rs.format.z16, color_format=rs.format.bgr8
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_format = depth_format
        self.color_format = color_format

        self._hardware_reset()

        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, self.depth_format, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, self.color_format, self.fps)
        self.config.enable_device(serial)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)

        self.camera_name = self.profile.get_device().get_info(rs.camera_info.name)

        self.align = rs.align(rs.stream.color)

        self.pc = rs.pointcloud()

        self.depth_frame = None
        self.depth_image = None
        self.color_frame = None
        self.color_image = None

        self.vertices = None
        self.colors = None

        self.color_intrinsic = None
        self.c2d_rotation = None
        self.c2d_translation = None
        self.depth_intrinsic = None
        self.d2c_rotation = None
        self.d2c_translation = None
        self.depth_scale = None

        self.get_camera_info_and_parameters(print_flag=False)

    def _hardware_reset(self):
        """Hardware-reset the device to clear any stale USB state."""
        ctx = rs.context()
        for dev in ctx.devices:
            if dev.get_info(rs.camera_info.serial_number) == self.serial:
                print(f"[RealSenseManager {self.serial}] Hardware-resetting device...")
                dev.hardware_reset()
                time.sleep(3)  # wait for device to re-enumerate on USB
                return
        print(f"[RealSenseManager {self.serial}] WARNING: device not found for hardware reset")

    def get_camera_info_and_parameters(
        self, print_flag=True
    ) -> tuple[rs.intrinsics, np.ndarray, np.ndarray, rs.intrinsics, np.ndarray, np.ndarray, float]:
        """
        Get camera intrinsic and extrinsic parameters.

        Return:
        -----------
            color_intr: Color Camera Intrinsics.
            c2d_rotation: Rotation from Color Camera to Depth Camera.
            c2d_translation: Translation from Color Camera to Depth Camera.
            depth_intr: Depth Camera Intrinsics.
            d2c_rotation: Rotation from Depth Camera to Color Camera.
            d2c_translation: Translation from Depth Camera to Color Camera.
            depth_scale: value to meter
        """

        # get stream
        color_stream = self.profile.get_stream(rs.stream.color)
        depth_stream = self.profile.get_stream(rs.stream.depth)

        # get stream profile and parameters
        color_intr = color_stream.as_video_stream_profile().get_intrinsics()
        c2d_extr = color_stream.as_video_stream_profile().get_extrinsics_to(depth_stream)

        c2d_rotation = np.array(c2d_extr.rotation).reshape(3, 3).transpose()  # transpose due to Column-major
        c2d_translation = np.array(c2d_extr.translation)

        # get stream profile and parameters
        depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()
        d2c_extr = depth_stream.as_video_stream_profile().get_extrinsics_to(color_stream)

        d2c_rotation = np.array(d2c_extr.rotation).reshape(3, 3).transpose()  # transpose due to Column-major
        d2c_translation = np.array(d2c_extr.translation)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # print values
        if print_flag:
            print(f"{self.camera_name}")
            print(" Color Intrinsic")
            print(f"   fx: {color_intr.fx}, fy: {color_intr.fy}, ppx: {color_intr.ppx}, ppy: {color_intr.ppy}")
            print(f"   distortion: {color_intr.model.name} [{color_intr.coeffs}]")
            print(" Color to Depth Extrinsic")
            print(f"   R:{c2d_rotation}")
            print(f"   t:{c2d_translation}")
            print(" Depth Intrinsic")
            print(f"   fx: {depth_intr.fx}, fy: {depth_intr.fy}, ppx: {depth_intr.ppx}, ppy: {depth_intr.ppy}")
            print(f"   distortion: {depth_intr.model.name} [{depth_intr.coeffs}]")
            print(" Depth to Color Extrinsic")
            print(f"   R:{d2c_rotation}")
            print(f"   t:{d2c_translation}")
            print(" Depth_scale: ", depth_scale)

        # update member variables
        self.color_intrinsic = color_intr
        self.c2d_rotation = c2d_rotation
        self.c2d_translation = c2d_translation
        self.depth_intrinsic = depth_intr
        self.d2c_rotation = d2c_rotation
        self.d2c_translation = d2c_translation
        self.depth_scale = depth_scale

        return color_intr, c2d_rotation, c2d_translation, depth_intr, d2c_rotation, d2c_translation, depth_scale

    def get_color_intrinsic(self):
        return self.color_intrinsic

    def get_color_extrinsic(self) -> tuple[np.ndarray, np.ndarray]:
        """Return rotation and translation matrices from color to depth"""
        return self.c2d_rotation, self.c2d_translation

    def get_depth_intrinsic(self):
        return self.depth_intrinsic

    def get_depth_extrinsic(self) -> tuple[np.ndarray, np.ndarray]:
        """Return rotation and translation matrices from depth to color"""
        return self.d2c_rotation, self.d2c_translation

    def get_depth_scale(self) -> float:
        """Return depth scale that converts the value of a depth pixel to meter"""
        return self.depth_scale

    def poll_frames(self):
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())

        return True

    def get_color_image(self):
        return self.color_image

    def get_depth_image(self):
        return self.depth_image

    def get_pointcloud_points_and_colors(self):
        self.pc.map_to(self.color_frame)
        self.points = self.pc.calculate(self.depth_frame)

        # Convert for Open3D
        self.vertices = np.asanyarray(self.points.get_vertices()).view(np.float32).reshape(-1, 3)
        self.vertices = self.vertices.astype(np.float64)
        self.colors = np.asanyarray(self.color_frame.get_data()).reshape(
            (self.color_frame.get_height(), self.color_frame.get_width(), 3)
        )
        self.colors = self.colors.reshape(-1, 3) / 255.0
        self.colors = self.colors.astype(np.float64)
        return self.vertices, self.colors

    def stop(self):
        self.pipeline.stop()
