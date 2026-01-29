import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skimage.feature import hog, local_binary_pattern
import joblib

from .utils import yolo_to_xyxy, xyxy_to_yolo, compute_iou


class PixelTemplateMatching:
    """
    PixelTemplateMatching is a simple object detection class that uses pixel-level template matching for detection.

    Attributes:
        threshold (float): Default similarity threshold for template matching.
        custom_threshold (dict): Optional per-class thresholds, e.g. {5: 0.98}.
        max_templates (int): Maximum number of templates to extract per class.
        nms_threshold (float): NMS IoU threshold.
        custom_threshold (dict): Optional dictionary specifying class-specific thresholds.

    Methods:
        get_threshold_for_class(cls_id): Returns the threshold for a specific class.
        train(dataset): Extracts templates from the training dataset.
        detect(img): Detects objects in the input image using template matching and NMS.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        max_templates_per_class: int = 500,
        nms_threshold: float = 0.5,
        custom_threshold: dict | None = None,
    ):
        self.threshold = threshold
        self.custom_threshold = custom_threshold or {}
        self.max_templates = max_templates_per_class
        self.nms_threshold = nms_threshold
        self.templates = {}

    def get_threshold_for_class(self, cls_id: int):
        """Get the detection threshold for a specific class."""
        return self.custom_threshold.get(cls_id, self.threshold)

    def train(self, dataset):
        """Extract templates from the training dataset."""
        print(
            f"Extracting up to {self.max_templates} templates per class, from {len(dataset)} images."
        )

        for i in range(dataset.nc):
            self.templates[i] = []

        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        for idx in tqdm(indices):
            if all(len(t) >= self.max_templates for t in self.templates.values()):
                break

            img, labels = dataset[idx]
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            h_img, w_img = img.shape

            for label in labels:
                cls_id = int(label[0])
                if len(self.templates[cls_id]) >= self.max_templates:
                    continue

                x1, y1, x2, y2 = yolo_to_xyxy(label, w_img, h_img)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)

                if (x2 - x1) > 15 and (y2 - y1) > 15:
                    template = img[y1:y2, x1:x2]
                    self.templates[cls_id].append(template)

        print("Training complete.")
        for k, v in self.templates.items():
            print(f"Class {k}: {len(v)} templates")

    def detect(self, img):
        """Detect objects in the input image using template matching."""
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        h_img, w_img = img_gray.shape

        all_boxes = []
        all_scores = []
        all_classes = []

        for cls_id, templates in self.templates.items():
            cls_threshold = self.get_threshold_for_class(cls_id)

            for template in templates:
                h_temp, w_temp = template.shape

                if h_temp >= h_img or w_temp >= w_img:
                    continue

                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

                y_loc, x_loc = np.where(res >= cls_threshold)

                if len(y_loc) == 0:
                    continue

                scores = res[y_loc, x_loc]
                n_hits = len(scores)

                w_arr = np.full(n_hits, w_temp)
                h_arr = np.full(n_hits, h_temp)
                cls_arr = np.full(n_hits, cls_id)

                boxes = np.column_stack((x_loc, y_loc, w_arr, h_arr))

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(cls_arr)

        if not all_boxes:
            return []

        all_boxes = np.concatenate(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)

        # Use the minimum threshold for NMS (safe lower bound)
        nms_score_threshold = min(
            [self.threshold] + list(self.custom_threshold.values())
        )

        indices = cv2.dnn.NMSBoxes(
            bboxes=all_boxes.tolist(),
            scores=all_scores.tolist(),
            score_threshold=nms_score_threshold,
            nms_threshold=self.nms_threshold,
        )

        predictions = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = all_boxes[i]
                cls_id = int(all_classes[i])

                # Final safety check (per-class threshold)
                if all_scores[i] < self.get_threshold_for_class(cls_id):
                    continue

                yolo_pred = xyxy_to_yolo(cls_id, x, y, x + w, y + h, w_img, h_img)
                predictions.append(yolo_pred)

        return predictions


class ImprovedPixelTemplateMatching:
    """
    An improved template matching class for object detection in images, particularly optimized for X-ray images.

    This class implements multi-scale and multi-rotation template matching with optional edge-based
    matching capabilities. It supports quality-based template filtering and non-maximum suppression
    for detection refinement.

    Attributes:
        threshold (float): Default confidence threshold for detection matching.
        custom_threshold (dict): Per-class custom threshold overrides mapping class IDs to thresholds.
        max_templates (int): Maximum number of templates to store per class.
        nms_threshold (float): IoU threshold for non-maximum suppression.
        templates (dict): Dictionary mapping class IDs to lists of grayscale template images.
        edge_templates (dict): Dictionary mapping class IDs to lists of edge-detected template images.
        scale_range (tuple): Min and max scale factors for multi-scale matching.
        scale_steps (int): Number of scale steps to use between scale_range bounds.
        scales (np.ndarray): Array of scale factors computed from scale_range and scale_steps.
        rotation_angles (list): List of rotation angles (in degrees) to try during matching.
        use_edge_matching (bool): Whether to use edge-based template matching.
        template_quality_threshold (float): Minimum quality score for a template to be accepted.

    Methods:
        get_threshold_for_class: Get the detection threshold for a specific class.
        train: Extract and store templates from a labeled dataset.
        detect: Detect objects in an image using stored templates.
    """

    def __init__(
        self,
        threshold: float = 0.90,
        max_templates_per_class: int = 300,
        nms_threshold: float = 0.4,
        custom_threshold: dict | None = None,
        scale_range: tuple = (0.7, 1.3),
        scale_steps: int = 5,
        rotation_angles: list = None,
        use_edge_matching: bool = True,
        template_quality_threshold: float = 0.15,
    ):
        self.threshold = threshold
        self.custom_threshold = custom_threshold or {}
        self.max_templates = max_templates_per_class
        self.nms_threshold = nms_threshold
        self.templates = {}
        self.edge_templates = {}

        # Multi-scale parameters
        self.scale_range = scale_range
        self.scale_steps = scale_steps
        self.scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

        # Rotation parameters
        if rotation_angles is None:
            self.rotation_angles = [0, 45, 90, 135, 180]
        else:
            self.rotation_angles = rotation_angles

        self.use_edge_matching = use_edge_matching
        self.template_quality_threshold = template_quality_threshold

    def get_threshold_for_class(self, cls_id: int):
        """Get the detection threshold for a specific class."""
        return self.custom_threshold.get(cls_id, self.threshold)

    def _compute_template_quality(self, template: np.ndarray):
        """Compute a quality score for a template based on variance and edge content."""
        variance = np.var(template) / (255.0**2)

        # Edge content using Sobel
        sobelx = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_score = np.mean(edge_magnitude) / 255.0

        # Quality score as weighted sum
        quality = 0.5 * variance + 0.5 * edge_score
        return quality

    def _extract_edges(self, img: np.ndarray):
        """Extract edges from the image using Canny edge detection."""
        # Apply slight blur to reduce noise in X-ray images
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def _rotate_template(self, template: np.ndarray, angle: float):
        """Rotate the template by a given angle."""
        h, w = template.shape
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix for new center
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(
            template, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def train(self, dataset):
        """Extract and store templates from the training dataset."""
        print(
            f"Extracting up to {self.max_templates} templates per class from {len(dataset)} images."
        )
        print(f"Using edge-based matching: {self.use_edge_matching}")

        # Initialize storage
        for i in range(dataset.nc):
            self.templates[i] = []
            if self.use_edge_matching:
                self.edge_templates[i] = []

        # Template extraction with quality filtering
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        template_qualities = {i: [] for i in range(dataset.nc)}

        for idx in tqdm(indices, desc="Extracting templates"):
            if all(len(t) >= self.max_templates for t in self.templates.values()):
                break

            img, labels = dataset[idx]
            img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            h_img, w_img = img_gray.shape

            if self.use_edge_matching:
                img_edges = self._extract_edges(img_gray)

            for label in labels:
                cls_id = int(label[0])
                if len(self.templates[cls_id]) >= self.max_templates:
                    continue

                # Convert YOLO format to pixel coordinates
                x_center, y_center, width, height = (
                    label[1],
                    label[2],
                    label[3],
                    label[4],
                )
                x_center_px = int(x_center * w_img)
                y_center_px = int(y_center * h_img)
                width_px = int(width * w_img)
                height_px = int(height * h_img)

                x1 = max(0, x_center_px - width_px // 2)
                y1 = max(0, y_center_px - height_px // 2)
                x2 = min(w_img, x_center_px + width_px // 2)
                y2 = min(h_img, y_center_px + height_px // 2)

                # Skip very small templates
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                # Extract template
                template = img_gray[y1:y2, x1:x2]

                # Quality check
                quality = self._compute_template_quality(template)
                if quality < self.template_quality_threshold:
                    continue

                # Store template with quality score
                self.templates[cls_id].append(template)
                template_qualities[cls_id].append(quality)

                if self.use_edge_matching:
                    edge_template = img_edges[y1:y2, x1:x2]
                    self.edge_templates[cls_id].append(edge_template)

        # Keep only the best templates if we have too many
        for cls_id in self.templates.keys():
            if len(self.templates[cls_id]) > self.max_templates:
                qualities = np.array(template_qualities[cls_id])
                top_indices = np.argsort(qualities)[-self.max_templates :]

                self.templates[cls_id] = [
                    self.templates[cls_id][i] for i in top_indices
                ]
                if self.use_edge_matching:
                    self.edge_templates[cls_id] = [
                        self.edge_templates[cls_id][i] for i in top_indices
                    ]

        print("\nTraining complete.")
        for k, v in self.templates.items():
            avg_quality = np.mean(template_qualities[k]) if template_qualities[k] else 0
            print(f"Class {k}: {len(v)} templates (avg quality: {avg_quality:.3f})")

    def _match_template_multiscale_rotation(
        self,
        img_gray: np.ndarray,
        template: np.ndarray,
        edge_img: np.ndarray = None,
        edge_template: np.ndarray = None,
        cls_threshold: float = 0.9,
    ):
        """Perform multi-scale and multi-rotation template matching."""
        h_img, w_img = img_gray.shape
        all_boxes = []
        all_scores = []

        # Try different rotations
        for angle in self.rotation_angles:
            # Rotate template
            if angle == 0:
                rot_template = template
                rot_edge_template = edge_template if edge_template is not None else None
            else:
                rot_template = self._rotate_template(template, angle)
                if self.use_edge_matching and edge_template is not None:
                    rot_edge_template = self._rotate_template(edge_template, angle)
                else:
                    rot_edge_template = None

            # Try different scales
            for scale in self.scales:
                h_temp, w_temp = rot_template.shape
                new_h, new_w = int(h_temp * scale), int(w_temp * scale)

                # Skip if scaled template is too large or too small
                if new_h >= h_img or new_w >= w_img or new_h < 15 or new_w < 15:
                    continue

                # Resize template
                scaled_template = cv2.resize(rot_template, (new_w, new_h))

                # Choose matching method
                if self.use_edge_matching and rot_edge_template is not None:
                    scaled_edge_template = cv2.resize(rot_edge_template, (new_w, new_h))
                    # Match on edges
                    res = cv2.matchTemplate(
                        edge_img, scaled_edge_template, cv2.TM_CCOEFF_NORMED
                    )
                else:
                    # Standard matching
                    res = cv2.matchTemplate(
                        img_gray, scaled_template, cv2.TM_CCOEFF_NORMED
                    )

                # Find matches above threshold
                y_loc, x_loc = np.where(res >= cls_threshold)

                if len(y_loc) == 0:
                    continue

                scores = res[y_loc, x_loc]

                # Create boxes
                for x, y, score in zip(x_loc, y_loc, scores):
                    all_boxes.append([x, y, new_w, new_h])
                    all_scores.append(score)

        return np.array(all_boxes) if all_boxes else np.array([]), np.array(
            all_scores
        ) if all_scores else np.array([])

    def detect(self, img):
        """Detect objects in the input image using stored templates."""
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        h_img, w_img = img_gray.shape

        if self.use_edge_matching:
            edge_img = self._extract_edges(img_gray)
        else:
            edge_img = None

        all_boxes = []
        all_scores = []
        all_classes = []

        for cls_id, templates in self.templates.items():
            cls_threshold = self.get_threshold_for_class(cls_id)

            for idx, template in enumerate(templates):
                edge_template = (
                    self.edge_templates[cls_id][idx] if self.use_edge_matching else None
                )

                boxes, scores = self._match_template_multiscale_rotation(
                    img_gray, template, edge_img, edge_template, cls_threshold
                )

                if len(boxes) > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_classes.append(np.full(len(boxes), cls_id))

        if not all_boxes:
            return []

        all_boxes = np.concatenate(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)

        # Apply NMS
        nms_score_threshold = min(
            [self.threshold] + list(self.custom_threshold.values())
        )

        indices = cv2.dnn.NMSBoxes(
            bboxes=all_boxes.tolist(),
            scores=all_scores.tolist(),
            score_threshold=nms_score_threshold,
            nms_threshold=self.nms_threshold,
        )

        predictions = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = all_boxes[i]
                cls_id = int(all_classes[i])
                score = all_scores[i]

                # Final per-class threshold check
                if score < self.get_threshold_for_class(cls_id):
                    continue

                # Convert to YOLO format
                x_center = (x + w / 2) / w_img
                y_center = (y + h / 2) / h_img
                width = w / w_img
                height = h / h_img

                predictions.append([cls_id, x_center, y_center, width, height, score])

        return predictions


class MetalMaskRandomForest:
    """
    MetalMaskRandomForest

    A machine learning model for detecting and classifying metallic regions in X-ray images.
    Uses a Random Forest classifier combined with HSV-based color detection to identify
    metallic objects and their characteristics.

    The model extracts features (aspect ratio, area, solidity) from detected metallic regions
    and classifies them based on training data.

    Attributes:
        model (RandomForestClassifier): The underlying Random Forest classifier with 100 estimators.
        img_size (int): The normalized image size used for feature normalization.
        lower_blue (np.ndarray): Lower HSV threshold for blue color detection [90, 50, 50].
        upper_blue (np.ndarray): Upper HSV threshold for blue color detection [140, 255, 255].
        lower_dark (np.ndarray): Lower HSV threshold for dark color detection [0, 0, 0].
        upper_dark (np.ndarray): Upper HSV threshold for dark color detection [180, 255, 100].
        max_box_coverage (float): Maximum normalized area threshold for bounding boxes (0.60).
        min_area_norm (float): Minimum normalized area threshold for bounding boxes (0.005).

    Methods:
        get_solidity(contour): Calculate the solidity of a contour (ratio of contour area to convex hull area).
        get_metal_mask(img): Generate a binary mask for metallic regions using HSV color thresholding.
        train(dataset): Train the Random Forest classifier on a dataset XRayDataset.
        detect(img): Detect and classify metallic objects in an image, returning normalized bounding boxes.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )

        # HSV & Detection Parameters
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_dark = np.array([0, 0, 0])
        self.upper_dark = np.array([180, 255, 100])
        self.max_box_coverage = 0.60
        self.min_area_norm = 0.005

    def get_solidity(self, contour):
        """Calculate the solidity of a contour."""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0
        return area / hull_area

    def get_metal_mask(self, img):
        """Generate a binary mask for metallic regions in the image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_dark = cv2.inRange(hsv, self.lower_dark, self.upper_dark)
        combined = cv2.bitwise_or(mask_blue, mask_dark)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    def train(self, dataset):
        """Train the Random Forest classifier on the provided dataset."""
        print(f"Training on {len(dataset)} images.")
        X_train = []
        y_train = []

        for img, labels in tqdm(dataset, desc="Building Feature Set"):
            if len(labels) == 0:
                continue

            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img.shape

            # Generate the mask once for the whole image
            full_mask = self.get_metal_mask(img)

            for box in labels:
                cls_id = int(box[0])
                # Denormalize box to pixels
                xc, yc, bw, bh = box[1], box[2], box[3], box[4]
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Crop the mask for this object
                obj_mask = full_mask[y1:y2, x1:x2]
                if obj_mask.size == 0:
                    continue

                # Find the contour of the object inside the box
                contours, _ = cv2.findContours(
                    obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    # If mask failed: assume a rectangle
                    solidity = 1.0
                else:
                    # Take largest contour in the crop
                    largest_cnt = max(contours, key=cv2.contourArea)
                    solidity = self.get_solidity(largest_cnt)

                # Features
                aspect_ratio = bw / (bh + 1e-6)
                area = bw * bh

                X_train.append([aspect_ratio, area, solidity])
                y_train.append(cls_id)

        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
            print("Training Complete.")

    def detect(self, img):
        """Detect and classify metallic objects in the input image."""
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        binary_mask = self.get_metal_mask(img)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        predictions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Normalize
            norm_w = w / img.shape[1]
            norm_h = h / img.shape[0]
            norm_area = norm_w * norm_h

            # Filters
            if norm_area < self.min_area_norm or norm_area > self.max_box_coverage:
                continue

            # Features
            aspect_ratio = norm_w / (norm_h + 1e-6)
            solidity = self.get_solidity(cnt)

            # Classification
            features = [[aspect_ratio, norm_area, solidity]]
            pred_class = int(self.model.predict(features)[0])

            norm_xc = (x + w / 2) / img.shape[1]
            norm_yc = (y + h / 2) / img.shape[0]
            predictions.append([pred_class, norm_xc, norm_yc, norm_w, norm_h])

        return predictions


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
        train(dataset, validate): Train both stage classifiers on the provided dataset.
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
        self._is_trained = False

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

    def train(self, dataset, validate: bool = True):
        """Train both stage classifiers on the provided dataset."""
        print(f"Training TwoStageDetector on {len(dataset)} images.")

        X_objects, y_objects, X_background = [], [], []
        class_stats = {
            i: {"areas": [], "aspect_ratios": [], "solidities": [], "intensities": []}
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
                    class_stats[cls_id]["intensities"].append(np.mean(roi_gray) / 255.0)
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

        # Stage 1: Object vs Background
        print("STAGE 1: Training Object vs Background classifier")

        n_background = min(len(X_background), self.max_reject_samples)
        bg_indices = np.random.choice(len(X_background), n_background, replace=False)
        X_bg_sampled = X_background[bg_indices]

        X_stage1 = np.vstack([X_objects, X_bg_sampled])
        y_stage1 = np.concatenate(
            [np.ones(len(X_objects)), np.zeros(len(X_bg_sampled))]
        )

        X_stage1_scaled = self.scaler1.fit_transform(X_stage1)
        if validate:
            scores = cross_val_score(
                self.stage1_classifier, X_stage1_scaled, y_stage1, cv=5
            )
            print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        self.stage1_classifier.fit(X_stage1_scaled, y_stage1)

        # Stage 2: Multi-class classification
        print("STAGE 2: Training Object Class classifier")

        X_stage2_scaled = self.scaler2.fit_transform(X_objects)
        if validate:
            scores = cross_val_score(
                self.stage2_classifier, X_stage2_scaled, y_objects, cv=5
            )
            print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        self.stage2_classifier.fit(X_stage2_scaled, y_objects)

        self._is_trained = True
        print("\nTraining complete.")

    def detect(self, img):
        """Detect objects in the input image using two-stage classification."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

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
