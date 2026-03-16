import numpy as np


# =========================================================
# EMA Filter
# =========================================================

class EMAFilter:
    """
    Exponential Moving Average filter
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.state = None

    def reset(self):
        self.state = None

    def __call__(self, keypoints: np.ndarray):

        keypoints = keypoints.astype(np.int32, copy=False)

        if self.state is None:
            self.state = keypoints.copy()
        else:
            self.state = self.alpha * keypoints + (1 - self.alpha) * self.state

        return self.state.copy()


# =========================================================
# One Euro Filter
# =========================================================

class OneEuroFilter:
    """
    One Euro Filter
    https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, min_cutoff=1.7, beta=0.5, d_cutoff=1.0, freq=30):

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq

        self.x_prev = None
        self.dx_prev = None

    def set_freq(self, freq):
        self.freq = max(freq, 1e-5)

    def alpha(self, cutoff):

        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)

        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):

        x = x.astype(np.float32)

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        dx = (x - self.x_prev) * self.freq

        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        alpha = self.alpha(cutoff)

        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


# =========================================================
# Kalman Filter
# =========================================================

class KalmanFilter2D:
    """
    Constant-velocity Kalman filter applied independently to each keypoint.
    State per keypoint: [x, y, vx, vy]
    """

    def __init__(self, process_var=0.09, measurement_var=5.0, freq=30.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.freq = freq
        self.state = None
        self.covariance = None

    def set_freq(self, freq):
        self.freq = max(freq, 1e-5)

    def __call__(self, keypoints: np.ndarray):
        z = np.asarray(keypoints, dtype=np.float32)[:, :2]
        num_points = z.shape[0]
        dt = 1.0 / max(self.freq, 1e-5)

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        observe = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        process_noise = np.eye(4, dtype=np.float32) * self.process_var
        measure_noise = np.eye(2, dtype=np.float32) * self.measurement_var
        identity = np.eye(4, dtype=np.float32)

        if self.state is None:
            self.state = np.zeros((num_points, 4), dtype=np.float32)
            self.state[:, :2] = z
            self.covariance = np.repeat(identity[None, :, :] * 10.0, num_points, axis=0)
            return z

        filtered = np.zeros_like(z)
        for idx in range(num_points):
            x_pred = transition @ self.state[idx]
            p_pred = transition @ self.covariance[idx] @ transition.T + process_noise

            innovation = z[idx] - observe @ x_pred
            innovation_cov = observe @ p_pred @ observe.T + measure_noise
            kalman_gain = p_pred @ observe.T @ np.linalg.inv(innovation_cov)

            x_new = x_pred + kalman_gain @ innovation
            p_new = (identity - kalman_gain @ observe) @ p_pred

            self.state[idx] = x_new
            self.covariance[idx] = p_new
            filtered[idx] = x_new[:2]

        return filtered


# =========================================================
# Multi-hand Filter Manager
# =========================================================

class MultiHandFilter:
    """
    Maintain filter per hand
    """

    def __init__(
        self,
        filter_type="none",
        max_match_distance=120.0,
        max_missing_frames=15,
        confirm_frames=3,
    ):

        self.filter_type = filter_type
        self.filters = {}
        self.track_centers = {}
        self.track_missing = {}
        self.track_hits = {}
        self.track_confirmed = {}
        self.next_track_id = 0
        self.max_match_distance = max_match_distance
        self.max_missing_frames = max_missing_frames
        self.confirm_frames = confirm_frames

    def create_filter(self):

        if self.filter_type == "ema":
            return EMAFilter()

        if self.filter_type == "oneeuro":
            return OneEuroFilter()

        if self.filter_type == "kalman":
            return KalmanFilter2D()

        return None

    def get_filter(self, hand_id):

        if hand_id not in self.filters:
            self.filters[hand_id] = self.create_filter()

        return self.filters[hand_id]

    def _compute_center(self, keypoints):

        pts = np.asarray(keypoints, dtype=np.float32)
        if pts.size == 0:
            return np.zeros(2, dtype=np.float32)

        return pts[:, :2].mean(axis=0)

    def _create_track(self, center, confirmed=False):

        track_id = f"hand_{self.next_track_id}"
        self.next_track_id += 1
        self.track_centers[track_id] = center
        self.track_missing[track_id] = 0
        self.track_hits[track_id] = 1
        self.track_confirmed[track_id] = confirmed
        self.get_filter(track_id)
        return track_id

    def _remove_track(self, track_id):

        self.filters.pop(track_id, None)
        self.track_centers.pop(track_id, None)
        self.track_missing.pop(track_id, None)
        self.track_hits.pop(track_id, None)
        self.track_confirmed.pop(track_id, None)

    def _assign_tracks(self, predictions):

        if not predictions:
            stale_ids = list(self.track_missing.keys())
            for track_id in stale_ids:
                self.track_missing[track_id] += 1
                if self.track_missing[track_id] > self.max_missing_frames:
                    self._remove_track(track_id)
            return []

        had_confirmed_tracks = any(self.track_confirmed.values())
        centers = [self._compute_center(keypoints) for keypoints, _ in predictions]
        unmatched_tracks = set(self.track_centers.keys())
        assignments = [None] * len(predictions)

        candidate_pairs = []
        for pred_idx, center in enumerate(centers):
            for track_id in unmatched_tracks:
                dist = np.linalg.norm(center - self.track_centers[track_id])
                candidate_pairs.append((dist, pred_idx, track_id))

        for dist, pred_idx, track_id in sorted(candidate_pairs, key=lambda x: x[0]):
            if dist > self.max_match_distance:
                continue
            if assignments[pred_idx] is not None or track_id not in unmatched_tracks:
                continue

            assignments[pred_idx] = track_id
            unmatched_tracks.remove(track_id)

        for pred_idx, center in enumerate(centers):
            if assignments[pred_idx] is None:
                assignments[pred_idx] = self._create_track(
                    center,
                    confirmed=not had_confirmed_tracks,
                )

        for track_id in list(unmatched_tracks):
            self.track_missing[track_id] += 1
            if self.track_missing[track_id] > self.max_missing_frames:
                self._remove_track(track_id)

        for pred_idx, track_id in enumerate(assignments):
            self.track_centers[track_id] = centers[pred_idx]
            self.track_missing[track_id] = 0
            self.track_hits[track_id] = self.track_hits.get(track_id, 0) + 1
            if not self.track_confirmed.get(track_id, False):
                if self.track_hits[track_id] >= self.confirm_frames:
                    self.track_confirmed[track_id] = True

        return assignments

    def update_freq(self, fps):

        for f in self.filters.values():

            if isinstance(f, OneEuroFilter):
                f.set_freq(fps)
            elif isinstance(f, KalmanFilter2D):
                f.set_freq(fps)

    def apply(self, predictions):

        outputs = []
        track_ids = self._assign_tracks(predictions)

        for (keypoints, label), hand_id in zip(predictions, track_ids):
            if not self.track_confirmed.get(hand_id, False):
                continue

            filt = self.get_filter(hand_id)

            if filt is not None:
                keypoints = filt(keypoints)

            outputs.append((keypoints.astype(np.int32), label))

        return outputs
