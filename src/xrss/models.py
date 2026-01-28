import cv2
import numpy as np
from tqdm import tqdm

from .utils import yolo_to_xyxy, xyxy_to_yolo


class TemplateMatching:
    def __init__(self, threshold=0.90, max_templates_per_class=500, iou_threshold=0.3):
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
