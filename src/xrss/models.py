import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from .utils import yolo_to_xyxy, xyxy_to_yolo


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
            print(
                f"Feature Importance (Aspect Ratio, Area, Solidity): {self.model.feature_importances_}"
            )

    def detect(self, img):
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


class PixelTemplateMatching:
    """
    PixelTemplateMatching is a simple object detection class that uses pixel-level template matching for detection.

    Attributes:
        threshold (float): Default similarity threshold for template matching.
        custom_threshold (dict): Optional per-class thresholds, e.g. {5: 0.98}.
        max_templates (int): Maximum number of templates to extract per class.
        nms_threshold (float): NMS IoU threshold.
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

    def get_threshold_for_class(self, cls_id: int) -> float:
        """Return class-specific threshold if provided, else default."""
        return self.custom_threshold.get(cls_id, self.threshold)

    def train(self, dataset):
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
