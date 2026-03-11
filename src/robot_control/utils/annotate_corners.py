from typing import List, Tuple, Dict
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use Tk backend (avoids Qt plugin conflicts)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def order_corners_ccw(corners: List[List[int]]) -> List[List[int]]:
    """Reorder 4 corners into counter-clockwise winding order around their centroid."""
    pts = np.array(corners, dtype=np.float64)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return [corners[i] for i in order]

def identify_ROIs(
    path: str,
    obj_list: List[str]
) -> Dict[str, List[Tuple[int,int]]]:
    """
    Let the user pick 4 corners for each object in obj_list via matplotlib clicks.
    Returns a dict mapping each object name to its 4 corner coords.
    """
    # Load once
    img = mpimg.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    rois: Dict[str, List[Tuple[int,int]]] = {}

    for obj in obj_list:
        corners: List[Tuple[int,int]] = []
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.set_title(f"Click 4 corners for '{obj}'\n(Press right‑click or any key to finish when done)")
        plt.axis('off')

        def onclick(event):
            # Only respond to left clicks inside the axes
            if event.inaxes is not ax or event.button != 1:
                return
            x, y = int(event.xdata), int(event.ydata)
            corners.append([x, y])
            ax.plot(x, y, 'ro')
            fig.canvas.draw()
            print(f"[{obj}] Picked corner #{len(corners)}: ({x}, {y})")
            # Auto‑disconnect if we've got 4
            if len(corners) == 4:
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)

        # Connect and show
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        if len(corners) != 4:
            raise RuntimeError(f"Object '{obj}': expected 4 corners, got {len(corners)}")

        corners = order_corners_ccw(corners)
        rois[obj] = corners
        print(f"[{obj}] Final corners (CCW ordered): {corners}")

    return rois

def main():
    import argparse, json as _json
    parser = argparse.ArgumentParser(description="Annotate ROI corners for FoundationPose")
    parser.add_argument("fp_dir", type=str, help="foundation_pose_dir (e.g. logs/foundation_pose_dir/gearmesh)")
    args = parser.parse_args()

    import os
    image_path = os.path.join(args.fp_dir, "detect_roi.jpg")
    corners = identify_ROIs(
        path=image_path,
        obj_list=["fixed_asset", "held_asset"],
    )
    out_path = os.path.join(args.fp_dir, "corners.json")
    with open(out_path, "w") as f:
        _json.dump(corners, f, indent=2)
    print(f"Saved corners to {out_path}")

if __name__ == "__main__":
    main()