import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from .utils import yolo_to_xyxy, xyxy_to_yolo


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
