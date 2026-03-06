import argparse
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.serialization
from ultralytics.nn.tasks import PoseModel
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
import ultralytics.nn.modules.head
import ultralytics.utils
import ultralytics.utils.loss
import ultralytics.utils.tal
import dill

torch.serialization.add_safe_globals([
    PoseModel,
    dill._dill._load_type,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.C3,
    ultralytics.nn.modules.block.C2,
    ultralytics.nn.modules.block.SPPF,
    ultralytics.nn.modules.head.Detect,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.container.ModuleList,
    ultralytics.nn.modules.block.Bottleneck,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    ultralytics.nn.modules.conv.Concat,
    ultralytics.nn.modules.head.Pose,
    ultralytics.nn.modules.block.DFL,
    getattr,
    ultralytics.utils.IterableSimpleNamespace,
    ultralytics.utils.loss.v8PoseLoss,
    torch.nn.modules.loss.BCEWithLogitsLoss,
    ultralytics.utils.tal.TaskAlignedAssigner,
    ultralytics.nn.tasks.DetectionModel,
    slice,
    range,
    tuple,
    ultralytics.utils.loss.BboxLoss,
    ultralytics.utils.loss.KeypointLoss
])

# -----------------------------
# Shared hand connections (21 keypoints)
# Same topology as MediaPipe Hands
# -----------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky + palm edge
]


def draw_hand_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    label: str = "",
    draw_bbox: bool = False,
) -> np.ndarray:
    """
    keypoints: (21, 2) or (21, 3), xy in pixel coordinates
    """
    if keypoints is None or len(keypoints) == 0:
        return image

    kpts = np.asarray(keypoints).copy()
    if kpts.shape[1] >= 2:
        xy = kpts[:, :2].astype(np.int32)
    else:
        return image

    # Draw connections
    # middle three fingers (index, middle, ring)
    RED_CONNECTIONS = {
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16)
    }

    for i, j in HAND_CONNECTIONS:
        x1, y1 = xy[i]
        x2, y2 = xy[j]

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:

            if (i, j) in RED_CONNECTIONS:
                color = (0, 0, 255)  # 紅色
            else:
                color = (0, 255, 0)  # 綠色

            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    # Draw joints
    for idx, (x, y) in enumerate(xy):
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

    # Draw bbox + label
    if draw_bbox:
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if label:
            cv2.putText(
                image,
                label,
                (x_min, max(y_min - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
    return image


class MediaPipeHandTracker:
    def __init__(self, max_num_hands=2, min_detection_conf=0.5, min_tracking_conf=0.5):
        import mediapipe as mp

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def predict(self, bgr_image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Return: list of (keypoints_xy, label)
        keypoints_xy: (21, 2)
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        outputs = []
        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or [None] * len(results.multi_hand_landmarks)

            h, w = bgr_image.shape[:2]
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, handedness_list):
                pts = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append([x, y])

                label = ""
                if handedness is not None and handedness.classification:
                    label = handedness.classification[0].label  # "Left" / "Right"

                outputs.append((np.array(pts, dtype=np.int32), label))

        return outputs

    def close(self):
        self.hands.close()


class WiLoRMiniHandTracker:
    def __init__(self):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.pipe = WiLorHandPose3dEstimationPipeline(
            device=self.device,
            dtype=self.dtype,
            verbose=False,
        )

    def predict(self, bgr_image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        WiLoR-mini expects RGB image in example usage.
        Return: list of (keypoints_xy, label)
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        outputs = self.pipe.predict(rgb)

        results = []
        for out in outputs:
            # Based on public usage examples / downstream code:
            # out.keys(): hand_bbox, is_right, wilor_preds
            # wilor_preds["pred_keypoints_2d"] shape is reported as (1, 21, 2) or (1, 21, 3)
            pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"]
            pred_keypoints_2d = np.asarray(pred_keypoints_2d)

            # Normalize to (21, 2)
            if pred_keypoints_2d.ndim == 3:
                pred_keypoints_2d = pred_keypoints_2d[0]
            pred_keypoints_2d = pred_keypoints_2d[:, :2]

            is_right = out.get("is_right", None)
            label = "Right" if bool(is_right) else "Left"

            results.append((pred_keypoints_2d.astype(np.int32), label))

        return results

    def close(self):
        pass


def build_tracker(backend: str):
    if backend == "mediapipe":
        return MediaPipeHandTracker()
    elif backend == "wilor-mini":
        return WiLoRMiniHandTracker()
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "wilor-mini"],
        help="Choose hand tracking backend.",
    )
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=640, help="Camera width.")
    parser.add_argument("--height", type=int, default=360, help="Camera height.")
    parser.add_argument("--show_fps", action="store_true", help="Show FPS on screen.")
    parser.add_argument("--flip", action="store_true", help="Flip image horizontally.")
    args = parser.parse_args()

    tracker = build_tracker(args.backend)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # -------- video writer --------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(
        "hand_skeleton_output.mp4",
        fourcc,
        10,
        (args.width, args.height)
    )

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        start = time.time()
        predictions = tracker.predict(frame)
        infer_time = time.time() - start

        vis = frame.copy()
        print(predictions)
        for keypoints, label in predictions:
            vis = draw_hand_skeleton(vis, keypoints, label=label, draw_bbox=True)

        if args.show_fps:
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                vis,
                f"Backend: {args.backend}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"FPS: {fps:.2f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"Infer: {infer_time*1000:.1f} ms",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Hand Skeleton Realtime", vis)
        video_out.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:  # ESC or q
            break

    tracker.close()
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()