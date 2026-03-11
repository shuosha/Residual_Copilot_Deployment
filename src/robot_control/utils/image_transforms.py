#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F  # noqa: N812

class InferenceTransforms(Transform):
    def __init__(self, cfg: dict, need_resize: bool=False, pusht: bool=False) -> None:
        super().__init__()
        # Inference transforms are identity transforms, i.e. no changes to the input.
        crop_size = cfg["tfs"]["crop"]["kwargs"]["size"]
        downsample_size = cfg["downsample"]["resize"]["kwargs"]["size"]
        self.crop = v2.CenterCrop(size=crop_size)
        self.downsample = v2.Resize(size=downsample_size)

        self.need_resize = need_resize
        self.resize = v2.Resize(size=downsample_size)

        print(f"Using inference transforms: {self.crop}, {self.downsample}")

        self.pusht = pusht
        if pusht:
            from PIL import Image
            import numpy as np, cv2

            self.device="cuda:0"
            goal_image_path = "experiments/log/data/20250825_pusht_50_v2_processed/debug/pushT_goal.png"
            t_mask_npz = "experiments/log/data/20250825_pusht_50_v2_processed/debug/T_mask.npz"

            # Load goal image to tensor [1,C,H,W]
            goal = np.array(Image.open(goal_image_path).convert("RGB")).astype(np.float32)/255.0
            if goal.shape[:2] != (480, 848):
                goal = cv2.resize(goal, (848, 480), interpolation=cv2.INTER_LINEAR)

            goal_yellow = goal.copy()
            goal_yellow[..., 0] = np.clip(goal_yellow[..., 0] * 1.2, 0.0, 1.0)  # boost R
            goal_yellow[..., 1] = np.clip(goal_yellow[..., 1] * 1.2, 0.0, 1.0)  # boost G
            goal_yellow[..., 2] *= 0.7  # reduce B

            goal_torch = torch.from_numpy(goal_yellow).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
            self.goal = goal_torch.to(self.device)

            # Load T mask to tensor [1,3,H,W]
            tmask = np.load(t_mask_npz)["mask"].astype(np.float32)  # (480,848)
            assert tmask.shape == (480,848), f"mask must be 480x848, got {tmask.shape}"
            t3 = np.repeat(tmask[...,None], 3, axis=2).astype(np.float32)  # (H,W,3)
            t3 = torch.from_numpy(t3).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
            self.t3 = t3.to(self.device)
            self.one_minus_t3 = (1.0 - self.t3)

            print("Overlaying T_goal")
    def forward(self, x: Any, overlay=False, pack_sloth=False) -> Any:
        if self.pusht:
            assert pack_sloth == False, "Cannot be both pusht and pack_sloth"
            if overlay:
                blended = 0.5 * self.goal + 0.5 * x
                x = blended * self.t3 + x * self.one_minus_t3
            x = x[:, :, :, 184:664]  # crop width to 480
            if overlay:
                x = x[:, :, 100:-100, :-40]  # further crop height to 280, width to 440

        elif pack_sloth:
            x = x[:, :, :, 100:-100]

        if self.need_resize:
            x = self.resize(x)
        x = self.crop(x)
        x = self.downsample(x)
        
        return x