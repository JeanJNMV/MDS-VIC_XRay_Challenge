import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skimage.feature import hog, local_binary_pattern

from .utils import yolo_to_xyxy, xyxy_to_yolo, compute_iou


class TwoStageDetector:
    """
    Two-stage object detector using Random Forest classifiers.

    Stage 1: Binary classification (object vs background) on region proposals.
    Stage 2: Multi-class classification on detected objects.

    Attributes:
        nc (int): Number of classes.
        max_reject_samples (int): Maximum background samples for training.
        stage1_threshold (float): Confidence threshold for stage 1.
        stage2_threshold (float): Default confidence threshold for stage 2.
        class_thresholds (dict): Per-class confidence thresholds for stage 2.
        stage2_class_weights (dict): Per-class weights for stage 2 classifier.

    Methods:
        train(dataset): Train both stage classifiers on the provided dataset.
        detect(img): Detect objects in the input image using two-stage classification.
    """

    def __init__(
        self,
        nc: int = 6,
        max_reject_samples: int = 15000,
        stage1_threshold: float = 0.45,
        n_estimators: int = 200,
        class_thresholds: dict | None = None,
        stage2_class_weights: dict | None = None,
    ):
        self.nc = nc
        self.max_reject_samples = max_reject_samples
        self.stage1_threshold = stage1_threshold

        # Proposal generator parameters
        self._min_area = 200
        self._max_area_ratio = 0.7
        self._nms_threshold = 0.5
        self._max_proposals = 100

        # Feature extractor parameters
        self._roi_size = (64, 64)
        self._lbp_radius = 3
        self._lbp_n_points = 24
        self._intensity_bins = 16

        # Stage 1: Binary classifier (Object vs Background)
        self.stage1_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Stage 2: Multi-class classifier
        self.stage2_class_weights = stage2_class_weights or {
            0: 1.2,
            1: 1.2,
            2: 1.0,
            3: 1.0,
            4: 1.0,
            5: 3.0,
        }
        self.stage2_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            class_weight=self.stage2_class_weights,
            random_state=42,
            n_jobs=-1,
        )

        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.class_constraints = {}
        self.class_thresholds = class_thresholds or {
            0: 0.70,
            1: 0.55,
            2: 0.40,
            3: 0.50,
            4: 0.40,
            5: 0.55,
        }
        self._train_features = None

    def _extract_features(self, roi: np.ndarray):
        """Extract HOG, edge, texture and shape features from an ROI."""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        gray_resized = cv2.resize(gray, self._roi_size)

        features = []
        features.extend(self._extract_hog_features(gray_resized))
        features.extend(self._extract_edge_features(gray_resized))
        features.extend(self._extract_texture_features(gray_resized))
        features.extend(self._extract_shape_features(gray))

        return np.array(features, dtype=np.float32)

    def _extract_hog_features(self, gray: np.ndarray):
        """Extract HOG features at multiple scales."""
        features = []
        for ppc in [(8, 8), (16, 16)]:
            try:
                hog_feat = hog(
                    gray,
                    orientations=9,
                    pixels_per_cell=ppc,
                    cells_per_block=(2, 2),
                    feature_vector=True,
                    block_norm="L2-Hys",
                )
                features.extend(
                    [
                        np.mean(hog_feat),
                        np.std(hog_feat),
                        np.max(hog_feat),
                        np.min(hog_feat),
                        np.percentile(hog_feat, 25),
                        np.percentile(hog_feat, 75),
                    ]
                )
            except Exception:
                features.extend([0.0] * 6)
        return features

    def _extract_edge_features(self, gray: np.ndarray):
        """Extract edge-based features."""
        features = []
        h, w = gray.shape
        total_pixels = h * w

        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / total_pixels)

        gray_float = np.float32(gray)
        corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        features.append(np.sum(corners > 0.01 * corners.max()) / total_pixels)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = (np.arctan2(sobely, sobelx) * 180 / np.pi + 180) % 180

        hist, _ = np.histogram(orientation, bins=9, range=(0, 180), weights=magnitude)
        hist = hist / (np.sum(hist) + 1e-6)
        features.extend(hist.tolist())
        features.extend([np.mean(magnitude), np.std(magnitude), np.max(magnitude)])

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=5
        )
        features.append((len(lines) if lines is not None else 0) / 100.0)

        return features

    def _extract_texture_features(self, gray: np.ndarray):
        """Extract intensity and texture features."""
        features = []
        features.extend(
            [
                np.mean(gray) / 255.0,
                np.std(gray) / 255.0,
                np.median(gray) / 255.0,
                np.min(gray) / 255.0,
                np.max(gray) / 255.0,
            ]
        )

        hist, _ = np.histogram(gray, bins=self._intensity_bins, range=(0, 256))
        hist = hist / (np.sum(hist) + 1e-6)
        features.extend(hist.tolist())

        try:
            lbp = local_binary_pattern(
                gray, P=self._lbp_n_points, R=self._lbp_radius, method="uniform"
            )
            n_bins = self._lbp_n_points + 2
            lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-6)
            features.extend(lbp_hist.tolist())
        except Exception:
            features.extend([0.0] * (self._lbp_n_points + 2))

        hist_nonzero = hist[hist > 0]
        features.append(-np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10)))
        features.append((np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0)
        features.append(np.var(cv2.Laplacian(gray, cv2.CV_64F)) / 10000.0)

        return features

    def _extract_shape_features(self, gray: np.ndarray):
        """Extract shape descriptors."""
        features = []
        h, w = gray.shape
        features.append(w / (h + 1e-6))
        features.append((w * h) / (64 * 64))

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            features.append(area / (hull_area + 1e-6))
            features.append(4 * np.pi * area / (perimeter**2 + 1e-6))

            x, y, bw, bh = cv2.boundingRect(cnt)
            features.append(area / (bw * bh + 1e-6))
            features.append(perimeter**2 / (area + 1e-6) / 100.0)

            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            if len(hull_indices) > 3 and len(cnt) > 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull_indices)
                    features.append(
                        np.sum(defects[:, 0, 3] > 1000) / 10.0
                        if defects is not None
                        else 0.0
                    )
                except Exception:
                    features.append(0.0)
            else:
                features.append(0.0)

            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.extend(hu_moments.tolist())

            if len(cnt) >= 5:
                try:
                    (cx, cy), (ma, MA), angle = cv2.fitEllipse(cnt)
                    features.extend([ma / (MA + 1e-6), angle / 180.0])
                except Exception:
                    features.extend([0.5, 0.0])
            else:
                features.extend([0.5, 0.0])
        else:
            features.extend([0.0] * 16)

        return features

    def _generate_proposals(self, img: np.ndarray):
        """Generate object proposals using multiple detection methods."""
        h, w = img.shape[:2]
        max_area = h * w * self._max_area_ratio
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        all_proposals = []
        all_proposals.extend(self._multi_threshold_intensity(gray, max_area))
        all_proposals.extend(self._percentile_based_detection(gray, max_area))
        all_proposals.extend(self._edge_detection(gray, max_area))
        all_proposals.extend(self._gradient_based_detection(gray, max_area))
        all_proposals.extend(self._clahe_detection(gray, max_area))
        all_proposals.extend(self._adaptive_threshold_detection(gray, max_area))
        all_proposals.extend(self._mser_detection(gray, max_area))
        all_proposals.extend(self._inverted_intensity_detection(gray, max_area))

        if all_proposals:
            all_proposals = self._apply_proposal_nms(all_proposals)

        if len(all_proposals) > self._max_proposals:
            all_proposals = sorted(
                all_proposals, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
            )[: self._max_proposals]

        return all_proposals

    def _extract_contours(self, binary: np.ndarray, max_area: float):
        """Extract bounding boxes from binary mask."""
        proposals = []
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self._min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 5 and h > 5:
                    proposals.append((x, y, x + w, y + h))
        return proposals

    def _multi_threshold_intensity(self, gray: np.ndarray, max_area: float):
        proposals = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for thresh in range(30, 180, 15):
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            proposals.extend(self._extract_contours(binary, max_area))
        return proposals

    def _percentile_based_detection(self, gray: np.ndarray, max_area: float):
        proposals = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for percentile in [5, 10, 15, 20, 25, 30]:
            thresh = np.percentile(gray, percentile)
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            proposals.extend(self._extract_contours(binary, max_area))
        return proposals

    def _edge_detection(self, gray: np.ndarray, max_area: float):
        proposals = []
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        for low, high in [(10, 50), (20, 80), (30, 100), (50, 150)]:
            edges = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilated = cv2.dilate(edges, kernel, iterations=3)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            proposals.extend(self._extract_contours(closed, max_area))
        return proposals

    def _gradient_based_detection(self, gray: np.ndarray, max_area: float):
        proposals = []
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        for thresh in [20, 40, 60]:
            _, binary = cv2.threshold(magnitude, thresh, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            binary = cv2.dilate(binary, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            proposals.extend(self._extract_contours(binary, max_area))
        return proposals

    def _clahe_detection(self, gray: np.ndarray, max_area: float):
        proposals = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for thresh in [40, 60, 80, 100]:
            _, binary = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            proposals.extend(self._extract_contours(binary, max_area))
        return proposals

    def _adaptive_threshold_detection(self, gray: np.ndarray, max_area: float):
        proposals = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for block_size in [11, 21, 31]:
            for c in [2, 5, 10]:
                binary = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    block_size,
                    c,
                )
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                proposals.extend(self._extract_contours(binary, max_area))
        return proposals

    def _mser_detection(self, gray: np.ndarray, max_area: float):
        proposals = []
        try:
            mser = cv2.MSER_create()
            mser.setMinArea(self._min_area // 2)
            mser.setMaxArea(int(max_area))
            mser.setDelta(5)
            regions, _ = mser.detectRegions(gray)
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                area = w * h
                if self._min_area < area < max_area and w > 5 and h > 5:
                    proposals.append((x, y, x + w, y + h))
        except Exception:
            pass
        return proposals

    def _inverted_intensity_detection(self, gray: np.ndarray, max_area: float):
        proposals = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for thresh in [180, 200, 220]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            proposals.extend(self._extract_contours(binary, max_area))
        return proposals

    def _apply_proposal_nms(self, boxes: List[Tuple]):
        """Apply NMS to remove duplicate proposals."""
        if not boxes:
            return []
        boxes_arr = np.array(boxes)
        areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (
            boxes_arr[:, 3] - boxes_arr[:, 1]
        )
        indices = np.argsort(-areas)

        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            if len(indices) == 1:
                break
            ious = np.array([compute_iou(boxes[i], boxes[j]) for j in indices[1:]])
            indices = indices[1:][ious < self._nms_threshold]

        return [boxes[i] for i in keep]

    def _learn_class_constraints(self, class_stats: Dict):
        """Learn size/shape constraints from training data."""
        for cls_id, stats in class_stats.items():
            if not stats["areas"]:
                continue
            areas = np.array(stats["areas"])
            aspects = np.array(stats["aspect_ratios"])
            solidities = np.array(stats["solidities"])
            intensities = np.array(stats["intensities"])

            self.class_constraints[cls_id] = {
                "area_min": np.percentile(areas, 5),
                "area_max": np.percentile(areas, 95),
                "aspect_min": np.percentile(aspects, 5),
                "aspect_max": np.percentile(aspects, 95),
                "solidity_min": np.percentile(solidities, 10),
                "intensity_max": np.percentile(intensities, 95),
            }

    def _check_constraints(
        self, cls_id: int, area: float, aspect: float, solidity: float, intensity: float
    ):
        """Check if a prediction satisfies class-specific constraints."""
        if cls_id not in self.class_constraints:
            return True
        c = self.class_constraints[cls_id]

        if cls_id == 5:
            if area < c["area_min"] * 0.5 or area > c["area_max"] * 1.5:
                return False
            return True

        if area < c["area_min"] * 0.7 or area > c["area_max"] * 1.3:
            return False
        if aspect < c["aspect_min"] * 0.6 or aspect > c["aspect_max"] * 1.4:
            return False
        if solidity < c["solidity_min"] * 0.6:
            return False
        return True

    def train(
        self,
        dataset,
        get_features: bool = True,
        stage1: bool = True,
        stage2: bool = True,
    ):
        """Train both stage classifiers on the provided dataset.

        Args:
            dataset: The training dataset.
            get_features: If True, extract features from the dataset and store them.
            stage1: If True, train the stage 1 (object vs background) classifier.
            stage2: If True, train the stage 2 (multi-class) classifier.
        """
        if get_features:
            print(f"Training TwoStageDetector on {len(dataset)} images.")

            X_objects, y_objects, X_background = [], [], []
            class_stats = {
                i: {
                    "areas": [],
                    "aspect_ratios": [],
                    "solidities": [],
                    "intensities": [],
                }
                for i in range(self.nc)
            }

            for idx in tqdm(range(len(dataset)), desc="Extracting features"):
                img, labels = dataset[idx]
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                gt_boxes = []
                for label in labels:
                    cls_id = int(label[0])
                    x1, y1, x2, y2 = yolo_to_xyxy(label, w, h)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue

                    gt_boxes.append((cls_id, x1, y1, x2, y2))
                    roi = img_bgr[y1:y2, x1:x2]

                    try:
                        features = self._extract_features(roi)
                        X_objects.append(features)
                        y_objects.append(cls_id)

                        box_w, box_h = x2 - x1, y2 - y1
                        area_norm = (box_w * box_h) / (w * h)
                        aspect = box_w / (box_h + 1e-6)

                        roi_gray = gray[y1:y2, x1:x2]
                        _, binary = cv2.threshold(
                            roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                        contours, _ = cv2.findContours(
                            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            hull_area = cv2.contourArea(cv2.convexHull(cnt))
                            solidity = cv2.contourArea(cnt) / (hull_area + 1e-6)
                        else:
                            solidity = 1.0

                        class_stats[cls_id]["areas"].append(area_norm)
                        class_stats[cls_id]["aspect_ratios"].append(aspect)
                        class_stats[cls_id]["solidities"].append(solidity)
                        class_stats[cls_id]["intensities"].append(
                            np.mean(roi_gray) / 255.0
                        )
                    except Exception:
                        continue

                proposals = self._generate_proposals(img_bgr)
                for px1, py1, px2, py2 in proposals:
                    if px2 - px1 < 10 or py2 - py1 < 10:
                        continue
                    is_background = all(
                        compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2)) <= 0.3
                        for _, gx1, gy1, gx2, gy2 in gt_boxes
                    )
                    if is_background:
                        roi = img_bgr[py1:py2, px1:px2]
                        try:
                            X_background.append(self._extract_features(roi))
                        except Exception:
                            continue

            X_objects = np.array(X_objects)
            y_objects = np.array(y_objects)
            X_background = np.array(X_background)

            print("\nData collected:")
            print(f"  Object samples: {len(X_objects)}")
            print(f"  Background samples: {len(X_background)}")
            print(f"  Object class distribution: {Counter(y_objects)}")

            self._learn_class_constraints(class_stats)

            # Store features for later use
            self._train_features = {
                "X_objects": X_objects,
                "y_objects": y_objects,
                "X_background": X_background,
            }

        # Check if features are available for training classifiers
        if (stage1 or stage2) and self._train_features is None:
            raise RuntimeError(
                "No features available. Call train with get_features=True first."
            )

        if stage1:
            X_objects = self._train_features["X_objects"]
            y_objects = self._train_features["y_objects"]
            X_background = self._train_features["X_background"]

            # Stage 1: Object vs Background
            print("STAGE 1: Training Object vs Background classifier")

            n_background = min(len(X_background), self.max_reject_samples)
            bg_indices = np.random.choice(
                len(X_background), n_background, replace=False
            )
            X_bg_sampled = X_background[bg_indices]

            X_stage1 = np.vstack([X_objects, X_bg_sampled])
            y_stage1 = np.concatenate(
                [np.ones(len(X_objects)), np.zeros(len(X_bg_sampled))]
            )

            X_stage1_scaled = self.scaler1.fit_transform(X_stage1)

            scores = cross_val_score(
                self.stage1_classifier, X_stage1_scaled, y_stage1, cv=5
            )
            print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            self.stage1_classifier.fit(X_stage1_scaled, y_stage1)

        if stage2:
            X_objects = self._train_features["X_objects"]
            y_objects = self._train_features["y_objects"]

            # Stage 2: Multi-class classification
            print("STAGE 2: Training Object Class classifier")

            X_stage2_scaled = self.scaler2.fit_transform(X_objects)

            scores = cross_val_score(
                self.stage2_classifier, X_stage2_scaled, y_objects, cv=5
            )
            print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            self.stage2_classifier.fit(X_stage2_scaled, y_objects)

        print("\nTraining complete.")

    def detect(self, img):
        """Detect objects in the input image using two-stage classification."""
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        proposals = self._generate_proposals(img_bgr)
        if not proposals:
            return []

        predictions = []
        for x1, y1, x2, y2 in proposals:
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            roi = img_bgr[y1:y2, x1:x2]
            try:
                features = self._extract_features(roi)

                # Stage 1: Filter background
                features_s1 = self.scaler1.transform([features])
                probs_s1 = self.stage1_classifier.predict_proba(features_s1)[0]
                object_prob = probs_s1[1] if len(probs_s1) > 1 else probs_s1[0]
                if object_prob < self.stage1_threshold:
                    continue

                # Stage 2: Classify object
                features_s2 = self.scaler2.transform([features])
                probs_s2 = self.stage2_classifier.predict_proba(features_s2)[0]
                pred_class = np.argmax(probs_s2)
                confidence = probs_s2[pred_class]

                if confidence < self.class_thresholds.get(pred_class, 0.5):
                    continue

                # Validate against learned constraints
                box_w, box_h = x2 - x1, y2 - y1
                area_norm = (box_w * box_h) / (w * h)
                aspect = box_w / (box_h + 1e-6)

                roi_gray = gray[y1:y2, x1:x2]
                _, binary = cv2.threshold(
                    roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    hull_area = cv2.contourArea(cv2.convexHull(cnt))
                    solidity = cv2.contourArea(cnt) / (hull_area + 1e-6)
                else:
                    solidity = 1.0

                if not self._check_constraints(
                    pred_class, area_norm, aspect, solidity, np.mean(roi_gray) / 255.0
                ):
                    continue

                yolo_pred = xyxy_to_yolo(pred_class, x1, y1, x2, y2, w, h)
                predictions.append(yolo_pred + [confidence])

            except Exception:
                continue

        predictions = self._class_specific_nms(predictions, w, h)
        return [pred[:5] for pred in predictions]

    def _class_specific_nms(self, predictions: List[List], img_w: int, img_h: int):
        """Apply NMS separately for each class."""
        if not predictions:
            return []

        class_preds = {}
        for pred in predictions:
            cls_id = int(pred[0])
            class_preds.setdefault(cls_id, []).append(pred)

        final_predictions = []
        for cls_id, preds in class_preds.items():
            boxes = []
            scores = []
            for pred in preds:
                x1, y1, x2, y2 = yolo_to_xyxy(pred[:5], img_w, img_h)
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(pred[5] if len(pred) > 5 else 0.5)

            indices = cv2.dnn.NMSBoxes(
                boxes, scores, score_threshold=0.1, nms_threshold=0.4
            )
            if len(indices) > 0:
                for i in indices.flatten():
                    final_predictions.append(preds[i])

        return final_predictions
