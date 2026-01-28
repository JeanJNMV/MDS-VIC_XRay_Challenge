import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from .utils import yolo_to_xyxy, xyxy_to_yolo


class TemplateMatching:
    def __init__(self, threshold=0.95, max_templates_per_class=500, iou_threshold=0.5):
        self.threshold = threshold
        self.max_templates = max_templates_per_class
        self.iou_threshold = iou_threshold
        self.templates = {}

    def train(self, dataset):
        print(f"Extracting up to {self.max_templates} templates per class.")
        # Reset templates
        for i in range(dataset.nc):
            self.templates[i] = []

        # Shuffle indices to get a random sampling of templates
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        for idx in tqdm(indices):
            # Stop if we have enough templates for all classes
            if all(len(t) >= self.max_templates for t in self.templates.values()):
                break

            img_pil, labels = dataset[idx]
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
            h_img, w_img = img_cv.shape

            for label in labels:
                cls_id = int(label[0])
                if len(self.templates[cls_id]) >= self.max_templates:
                    continue

                x1, y1, x2, y2 = yolo_to_xyxy(label, w_img, h_img)

                # Safety checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if (x2 - x1) > 15 and (y2 - y1) > 15:
                    template = img_cv[y1:y2, x1:x2]
                    self.templates[cls_id].append(template)

        print("Template extraction complete.")
        for k, v in self.templates.items():
            print(f"Class {k}: {len(v)} templates")

    def detect(self, img_pil):
        img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
        h_img, w_img = img_gray.shape

        all_boxes = []
        all_scores = []
        all_classes = []

        # Iterate over all templates
        for cls_id, templates in self.templates.items():
            for template in templates:
                h_temp, w_temp = template.shape

                # Skip if template is bigger than the image
                if h_temp >= h_img or w_temp >= w_img:
                    continue

                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

                y_loc, x_loc = np.where(res >= self.threshold)

                if len(y_loc) > 0:
                    scores = res[y_loc, x_loc]
                    n_hits = len(scores)

                    w_arr = np.full(n_hits, w_temp)
                    h_arr = np.full(n_hits, h_temp)
                    cls_arr = np.full(n_hits, cls_id)

                    current_boxes = np.column_stack((x_loc, y_loc, w_arr, h_arr))

                    all_boxes.append(current_boxes)
                    all_scores.append(scores)
                    all_classes.append(cls_arr)

        if not all_boxes:
            return []

        all_boxes = np.concatenate(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            bboxes=all_boxes.tolist(),
            scores=all_scores.tolist(),
            score_threshold=self.threshold,
            nms_threshold=self.iou_threshold,
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Retrieve the box in [x, y, w, h] format
                x, y, w, h = all_boxes[i]
                cls_id = all_classes[i]

                # Convert to YOLO format
                yolo_pred = xyxy_to_yolo(int(cls_id), x, y, x + w, y + h, w_img, h_img)
                results.append(yolo_pred)

        return results


class RandomForestDetector:
    def __init__(self, img_size=416):
        self.img_size = img_size
        # Switch to Random Forest with 'balanced' weights to fix the "All Hammer" bias
        self.model = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )
        self.is_trained = False

        # HSV & Detection Parameters
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_dark = np.array([0, 0, 0])
        self.upper_dark = np.array([180, 255, 100])
        self.max_box_coverage = 0.60
        self.min_area_norm = 0.005

    def get_solidity(self, contour):
        """
        Calculates Solidity: Ratio of Contour Area to Convex Hull Area.
        Perfect Rectangle = 1.0
        L-Shape (Gun) ~= 0.6 - 0.7
        """
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0
        return area / hull_area

    def get_metal_mask(self, img_cv):
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_dark = cv2.inRange(hsv, self.lower_dark, self.upper_dark)
        combined = cv2.bitwise_or(mask_blue, mask_dark)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    def train(self, dataset):
        print(f"Training on {len(dataset)} images (Extraction will take a moment)...")
        X_train = []
        y_train = []

        # We must open images now to calculate Solidity for the Ground Truth
        for img_pil, labels in tqdm(dataset, desc="Building Feature Set"):
            if len(labels) == 0:
                continue

            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            h, w, _ = img_cv.shape

            # Generate the mask once for the whole image
            full_mask = self.get_metal_mask(img_cv)

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

                # Find the contour of the object INSIDE the box
                contours, _ = cv2.findContours(
                    obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    # Fallback if mask failed: assume a rectangle
                    solidity = 1.0
                else:
                    # Take largest contour in the crop
                    largest_cnt = max(contours, key=cv2.contourArea)
                    solidity = self.get_solidity(largest_cnt)

                # FEATURES: [Aspect Ratio, Area, Solidity]
                aspect_ratio = bw / (bh + 1e-6)
                area = bw * bh

                X_train.append([aspect_ratio, area, solidity])
                y_train.append(cls_id)

        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            print(f"Training Complete. Learned {len(X_train)} samples.")
            # Print feature importance
            print(
                f"Feature Importance (Aspect Ratio, Area, Solidity): {self.model.feature_importances_}"
            )

    def detect(self, img_pil):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        binary_mask = self.get_metal_mask(img_cv)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        predictions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Normalize
            norm_w = w / self.img_size
            norm_h = h / self.img_size
            norm_area = norm_w * norm_h

            # Filters
            if norm_area < self.min_area_norm or norm_area > self.max_box_coverage:
                continue

            # Features
            aspect_ratio = norm_w / (norm_h + 1e-6)
            solidity = self.get_solidity(cnt)

            # Classification
            pred_class = 0
            if self.is_trained:
                features = [[aspect_ratio, norm_area, solidity]]
                pred_class = int(self.model.predict(features)[0])

            norm_xc = (x + w / 2) / self.img_size
            norm_yc = (y + h / 2) / self.img_size
            predictions.append([pred_class, norm_xc, norm_yc, norm_w, norm_h])

        return predictions
