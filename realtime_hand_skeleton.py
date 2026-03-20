import argparse
import csv
import json
from pathlib import Path
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


from filter import MultiHandFilter
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

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


def compute_hand_bbox(keypoints: np.ndarray) -> Tuple[int, int, int, int]:
    pts = np.asarray(keypoints)
    xy = pts[:, :2].astype(np.int32)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def suppress_duplicate_hands(
    detections: List[Tuple[np.ndarray, str, float]],
    center_distance_thresh: float = 50.0,
    iou_thresh: float = 0.4,
) -> List[Tuple[np.ndarray, str]]:
    kept = []

    for keypoints, label, score in sorted(detections, key=lambda x: x[2], reverse=True):
        center = np.asarray(keypoints[:, :2], dtype=np.float32).mean(axis=0)
        bbox = compute_hand_bbox(keypoints)

        is_duplicate = False
        for kept_keypoints, kept_label, kept_score, kept_center, kept_bbox in kept:
            if label != kept_label:
                continue

            center_distance = np.linalg.norm(center - kept_center)
            overlap = bbox_iou(bbox, kept_bbox)
            if center_distance < center_distance_thresh or overlap > iou_thresh:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append((keypoints, label, score, center, bbox))

    return [(keypoints, label) for keypoints, label, _, _, _ in kept]


def pre_process_landmark(landmark_list: np.ndarray) -> List[float]:
    temp_landmark_list = np.asarray(landmark_list, dtype=np.float32).copy()

    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    temp_landmark_list[:, 0] = temp_landmark_list[:, 0] - base_x
    temp_landmark_list[:, 1] = temp_landmark_list[:, 1] - base_y

    flattened = temp_landmark_list.flatten().tolist()
    max_value = max(map(abs, flattened), default=0.0)
    if max_value == 0:
        return flattened

    return [value / max_value for value in flattened]


def classify_hand_sign(
    landmark_list: np.ndarray,
    keypoint_classifier: KeyPointClassifier,
    keypoint_classifier_labels: List[str],
) -> Tuple[str, float]:
    pre_processed_landmarks = pre_process_landmark(landmark_list)
    hand_sign_id, confidence = keypoint_classifier(pre_processed_landmarks)
    confidence = float(confidence)

    return keypoint_classifier_labels[hand_sign_id], confidence


def get_winning_hand(player_gesture: str) -> str:
    winning_map = {
        "Stone": "Paper",
        "Paper": "Scissors",
        "Scissors": "Stone",
    }
    return winning_map.get(player_gesture, "")


def get_losing_hand(player_gesture: str) -> str:
    losing_map = {
        "Stone": "Scissors",
        "Paper": "Stone",
        "Scissors": "Paper",
    }
    return losing_map.get(player_gesture, "")


def gesture_to_slug(gesture: str) -> str:
    slug_map = {
        "Paper": "paper",
        "Stone": "rock",
        "Scissors": "scissors",
    }
    return slug_map.get(gesture, "")


def extract_first_valid_hand_from_frame(frame_data) -> np.ndarray | None:
    for hand in frame_data:
        hand_array = np.asarray(hand, dtype=np.float32)
        if hand_array.shape[0] == 21 and hand_array[0][0] >= 0:
            return hand_array[:, :2]
    return None


def extract_first_valid_hand_from_json(json_path: Path) -> np.ndarray | None:
    with json_path.open(encoding="utf-8") as f:
        frames = json.load(f)

    for frame in frames:
        hand_array = extract_first_valid_hand_from_frame(frame)
        if hand_array is not None:
            return hand_array
    return None


def load_game_skeleton_animations(base_dir: Path) -> dict[str, list[np.ndarray]]:
    animations = {}
    for json_path in sorted(base_dir.glob("*_skeleton.json")):
        animation_key = json_path.stem.replace("_skeleton", "")

        with json_path.open(encoding="utf-8") as f:
            frames = json.load(f)

        gesture_frames = []
        for frame in frames:
            hand_array = extract_first_valid_hand_from_frame(frame)
            if hand_array is not None:
                gesture_frames.append(hand_array)

        if gesture_frames:
            animations[animation_key] = gesture_frames
    return animations


def normalize_hand_for_canvas(keypoints: np.ndarray, canvas_size: int = 360) -> np.ndarray:
    pts = np.asarray(keypoints, dtype=np.float32)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    size = np.maximum(max_xy - min_xy, 1.0)
    scale = min((canvas_size - 80) / size[0], (canvas_size - 100) / size[1])

    normalized = (pts - min_xy) * scale
    offset = np.array([
        (canvas_size - (size[0] * scale)) / 2.0,
        (canvas_size - (size[1] * scale)) / 2.0 + 25.0,
    ], dtype=np.float32)
    normalized += offset
    return normalized


def render_game_skeleton_frame(gesture: str, keypoints: np.ndarray, canvas_size: int = 360) -> np.ndarray:
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"Computer: {gesture}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 128, 0),
        2,
        cv2.LINE_AA,
    )

    normalized = normalize_hand_for_canvas(keypoints, canvas_size=canvas_size)

    return draw_hand_skeleton(
        canvas,
        normalized.astype(np.int32),
        draw_bbox=False,
    )


class GameSkeletonPlayer:
    def __init__(self, animations: dict[str, list[np.ndarray]]):
        self.animations = animations
        self.current_gesture = ""
        self.current_animation_key = ""
        self.frame_index = 0

    def set_gesture(self, gesture: str) -> None:
        if gesture != self.current_gesture:
            previous_slug = gesture_to_slug(self.current_gesture)
            next_slug = gesture_to_slug(gesture)
            transition_key = f"{previous_slug}_{next_slug}" if previous_slug and next_slug else ""

            self.current_gesture = gesture
            if transition_key and transition_key in self.animations:
                self.current_animation_key = transition_key
            else:
                self.current_animation_key = next_slug
            self.frame_index = 0

    def get_frame(self) -> np.ndarray | None:
        frames = self.animations.get(self.current_animation_key)
        if not frames:
            return None

        keypoints = frames[self.frame_index]
        if self.frame_index < len(frames) - 1:
            self.frame_index = min(self.frame_index + 3, len(frames) - 1)
        return render_game_skeleton_frame(self.current_gesture, keypoints)


class JitterEvaluator:
    def __init__(self, max_match_distance: float = 120.0):
        self.max_match_distance = max_match_distance
        self.prev_tracks = {}
        self.next_track_id = 0
        self.global_motion_sum = 0.0
        self.relative_jitter_sum = 0.0
        self.residual_sum = 0.0
        self.match_count = 0

    def _compute_center(self, keypoints: np.ndarray) -> np.ndarray:
        pts = np.asarray(keypoints, dtype=np.float32)
        return pts[:, :2].mean(axis=0)

    def _compute_relative(self, keypoints: np.ndarray, center: np.ndarray) -> np.ndarray:
        pts = np.asarray(keypoints, dtype=np.float32)[:, :2]
        return pts - center

    def update(self, predictions: List[Tuple[np.ndarray, str]]) -> None:
        if not predictions:
            self.prev_tracks = {}
            return

        current = []
        for keypoints, _ in predictions:
            center = self._compute_center(keypoints)
            relative = self._compute_relative(keypoints, center)
            current.append({
                "keypoints": keypoints,
                "center": center,
                "relative": relative,
            })

        unmatched_prev = set(self.prev_tracks.keys())
        assignments = [None] * len(current)
        candidate_pairs = []

        for curr_idx, item in enumerate(current):
            for track_id in unmatched_prev:
                prev_center = self.prev_tracks[track_id]["center"]
                dist = np.linalg.norm(item["center"] - prev_center)
                candidate_pairs.append((dist, curr_idx, track_id))

        for dist, curr_idx, track_id in sorted(candidate_pairs, key=lambda x: x[0]):
            if dist > self.max_match_distance:
                continue
            if assignments[curr_idx] is not None or track_id not in unmatched_prev:
                continue

            assignments[curr_idx] = track_id
            unmatched_prev.remove(track_id)

        next_tracks = {}
        for curr_idx, item in enumerate(current):
            track_id = assignments[curr_idx]
            if track_id is None:
                track_id = f"eval_{self.next_track_id}"
                self.next_track_id += 1
            else:
                prev_item = self.prev_tracks[track_id]
                global_motion = np.linalg.norm(item["center"] - prev_item["center"])
                relative_delta = item["relative"] - prev_item["relative"]
                relative_jitter = np.linalg.norm(relative_delta, axis=1).mean()

                self.global_motion_sum += float(global_motion)
                self.relative_jitter_sum += float(relative_jitter)
                self.residual_sum += float(max(relative_jitter - global_motion, 0.0))
                self.match_count += 1

            next_tracks[track_id] = item

        self.prev_tracks = next_tracks

    def summary(self) -> dict:
        if self.match_count == 0:
            return {
                "global_motion": 0.0,
                "relative_jitter": 0.0,
                "jitter_ratio": 0.0,
                "matched_pairs": 0,
            }

        avg_global_motion = self.global_motion_sum / self.match_count
        avg_relative_jitter = self.relative_jitter_sum / self.match_count
        jitter_ratio = avg_relative_jitter / max(avg_global_motion, 1e-6)
        return {
            "global_motion": avg_global_motion,
            "relative_jitter": avg_relative_jitter,
            "jitter_ratio": jitter_ratio,
            "matched_pairs": self.match_count,
        }


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
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        # (9,13),(13,14),(14,15),(15,16)
    }

    for i, j in HAND_CONNECTIONS:
        x1, y1 = xy[i]
        x2, y2 = xy[j]

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:

            if (i, j) in RED_CONNECTIONS:
                color = (0, 0, 255)
            else:
                color = (0, 0, 255)

            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    # Draw joints
    for idx, (x, y) in enumerate(xy):
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    # Draw bbox + label
    if draw_bbox:
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

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
    def __init__(self, max_num_hands=1, min_detection_conf=0.5, min_tracking_conf=0.5):
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
                score = 0.0
                if handedness is not None and handedness.classification:
                    label = handedness.classification[0].label  # "Left" / "Right"
                    score = float(handedness.classification[0].score)

                outputs.append((np.array(pts, dtype=np.int32), label, score))

        return suppress_duplicate_hands(outputs)

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


def resolve_video_io(args) -> Tuple[cv2.VideoCapture, cv2.VideoWriter | None, str | None, bool]:
    is_test_mode = bool(args.testmode)

    if is_test_mode:
        input_path = Path(args.testmode)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = input_path.with_name(f"{input_path.stem}_{args.filter}_{args.backend}_skeleton.mp4")
    else:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        width = args.width
        height = args.height
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = Path("hand_skeleton_output.mp4")

    video_out = None
    if not args.no_save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not video_out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")
    else:
        output_path = None

    return cap, video_out, (str(output_path) if output_path is not None else None), is_test_mode


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
    parser.add_argument(
        "--testmode",
        type=str,
        default="",
        help="Path to an input video. When set, process this video instead of webcam.",
    )
    parser.add_argument("--show_fps", action="store_true", help="Show FPS on screen.")
    parser.add_argument("--flip", action="store_true", help="Flip image horizontally.")
    parser.add_argument(
        "--eval_jitter",
        action="store_true",
        help="Evaluate global motion and relative jitter over the processed video.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="none",
        choices=["none", "ema", "oneeuro", "kalman"],
        help="Temporal smoothing filter"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="none",
        choices=["none", "gripper_win", "gripper_lose"],
        help="Game response mode: print the computer hand that wins or loses on gesture change.",
    )
    parser.add_argument(
        "--no_save_video",
        action="store_true",
        help="Disable saving the output skeleton video.",
        default=True,
    )
    args = parser.parse_args()

    tracker = build_tracker(args.backend)
    filter_manager = MultiHandFilter(args.filter)
    jitter_evaluator = JitterEvaluator() if args.eval_jitter else None
    cap, video_out, output_path, is_test_mode = resolve_video_io(args)

    prev_time = time.time()
    total_infer_time = 0.0
    frame_count = 0
    keypoint_classifier = KeyPointClassifier()
    previous_gesture = None
    current_computer_gesture = ""
    current_computer_frame = None
    game_skeleton_player = GameSkeletonPlayer(
        load_game_skeleton_animations(Path("game_skeleton"))
    )
    keypoint_label_path = Path("model/keypoint_classifier/keypoint_classifier_label.csv")
    with keypoint_label_path.open(encoding="utf-8-sig") as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    while True:
        ok, frame = cap.read()
        if not ok:
            if not is_test_mode:
                print("Failed to read frame from camera.")
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        start = time.time()
        predictions = tracker.predict(frame)
        # print("raw:", len(predictions))
        predictions = filter_manager.apply(predictions)
        # print("filtered:", len(predictions))
        
        infer_time = time.time() - start
        total_infer_time += infer_time
        frame_count += 1
        if jitter_evaluator is not None:
            jitter_evaluator.update(predictions)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        filter_manager.update_freq(fps)

        vis = frame.copy()
        current_gesture = None
        for keypoints, label in predictions:
            gesture_label, confidence = classify_hand_sign(
                keypoints,
                keypoint_classifier,
                keypoint_classifier_labels,
            )
            current_gesture = gesture_label
            display_label = gesture_label
            if label:
                display_label = f"{label} | {gesture_label}"
            if confidence > 0:
                display_label = f"{display_label} ({confidence:.2f})"

            vis = draw_hand_skeleton(
                vis,
                keypoints,
                label=display_label,
                draw_bbox=True,
            )

        if current_gesture != previous_gesture:
            computer_gesture = ""
            if args.game == "gripper_win":
                computer_gesture = get_winning_hand(current_gesture or "")
            elif args.game == "gripper_lose":
                computer_gesture = get_losing_hand(current_gesture or "")

            if computer_gesture:
                # game output
                print(f"Player: {current_gesture} -> Computer: {computer_gesture}")
                game_skeleton_player.set_gesture(computer_gesture)
            current_computer_gesture = computer_gesture
        previous_gesture = current_gesture

        if args.game != "none" and current_computer_gesture:
            current_computer_frame = game_skeleton_player.get_frame()
        else:
            current_computer_frame = None

        if args.show_fps:

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

        if args.game != "none" and current_computer_gesture:
            cv2.putText(
                vis,
                f"Computer: {current_computer_gesture}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if video_out is not None:
            video_out.write(vis)

        if not is_test_mode:
            cv2.imshow("Hand Skeleton Realtime", vis)
            if args.game != "none" and current_computer_frame is not None:
                cv2.imshow("Computer Gesture", current_computer_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:  # ESC or q
                break

    tracker.close()
    cap.release()
    if video_out is not None:
        video_out.release()
    cv2.destroyAllWindows()

    avg_infer_ms = (total_infer_time / frame_count * 1000.0) if frame_count else 0.0
    if output_path is not None:
        print(f"Output video saved to: {output_path}")
    else:
        print("Output video saving disabled.")
    print(f"Average inference time: {avg_infer_ms:.2f} ms over {frame_count} frames")
    if jitter_evaluator is not None:
        jitter_stats = jitter_evaluator.summary()
        print(
            "Average global motion: "
            f"{jitter_stats['global_motion']:.3f} px over {jitter_stats['matched_pairs']} matched frame pairs"
        )
        print(f"Average relative jitter: {jitter_stats['relative_jitter']:.3f} px")
        print(f"Jitter/global ratio: {jitter_stats['jitter_ratio']:.3f}")


if __name__ == "__main__":
    main()
