import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from pathlib import Path


def yolo_to_xyxy(box, img_w, img_h):
    """Convert a bounding box from YOLO format to [x1, y1, x2, y2] format."""
    _, xc, yc, w, h = box
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, img_w, img_h):
    """Convert a bounding box from [x1, y1, x2, y2] format to YOLO format."""
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cls, xc, yc, w, h]


def compute_iou(a, b):
    """Compute Intersection over Union (IoU) between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)


def evaluate_score(model, dataset):
    """Evaluate the model on the dataset and return the average IoU-based score."""
    frame_scores = []

    for i in tqdm(range(len(dataset))):
        img, gt_labels = dataset[i]
        img_h, img_w = img.shape[:2]

        pred_labels = model.detect(img)

        num_gt = len(gt_labels)
        num_pred = len(pred_labels)

        # Both empty
        if num_gt == 0 and num_pred == 0:
            frame_scores.append(1.0)
            continue

        # One empty
        if num_gt == 0 or num_pred == 0:
            frame_scores.append(0.0)
            continue

        total_iou = 0.0
        used_preds = set()

        # Iterate through ground truth boxes
        for gt in gt_labels:
            gt_cls = int(gt[0])
            gt_box = yolo_to_xyxy(gt, img_w, img_h)

            best_iou = 0.0
            best_match_idx = -1

            for p_idx, pred in enumerate(pred_labels):
                if p_idx in used_preds:
                    continue

                pred_cls = int(pred[0])

                # Only match if classes are the same
                if pred_cls == gt_cls:
                    pred_box = yolo_to_xyxy(pred, img_w, img_h)
                    iou = compute_iou(gt_box, pred_box)

                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = p_idx

            if best_match_idx != -1:
                total_iou += best_iou
                used_preds.add(best_match_idx)

        frame_score = total_iou / max(num_gt, num_pred)
        frame_scores.append(frame_score)

    return np.mean(frame_scores)


def compute_predictions_folder(
    model, dataset, output_folder="./predictions", replace=True
):
    """Compute predictions for all images in the dataset and save them to the output folder in YOLO format."""
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        preds = model.detect(img)

        # Determine output file path
        img_name = os.path.splitext(os.path.basename(dataset.img_files[i]))[0]
        pred_file = os.path.join(output_folder, img_name + ".txt")

        # Check if file exists
        if os.path.exists(pred_file) and not replace:
            print(f"File {pred_file} already exists.")
            continue

        # Write predictions
        if len(preds) > 0:
            np.savetxt(pred_file, preds, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])
        else:
            # Save empty file if no predictions
            open(pred_file, "w").close()


def yolo_to_submission_csv(yolo_dir, output_csv):
    """Convert YOLO format txt files to submission CSV."""
    data = []

    for txt_file in sorted(Path(yolo_dir).glob("*.txt")):
        frame_id = txt_file.stem  # filename without extension
        boxes = []

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    boxes.append([int(cls), cx, cy, w, h])

        data.append({"frame_id": frame_id, "bbs": str(boxes)})

    pd.DataFrame(data).to_csv(output_csv, index=False)
