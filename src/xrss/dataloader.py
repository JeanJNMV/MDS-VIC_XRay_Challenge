import os
import yaml
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import math


class XRayDataset:
    def __init__(self, yaml_file, split="train", img_size=416):
        # Load yaml file
        with open(yaml_file, "r") as f:
            cfg = yaml.safe_load(f)

        # Dataset parameters
        yaml_dir = os.path.dirname(yaml_file)
        self.root = yaml_dir
        self.split = split
        self.img_size = img_size
        self.mapping = {i: name for i, name in enumerate(cfg["names"])}
        self.nc = cfg["nc"]

        # Load images
        self.img_dir = os.path.join(self.root, cfg[split])
        self.img_files = sorted(
            [
                os.path.join(self.img_dir, f)
                for f in os.listdir(self.img_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
        )

        # Load labels if they exist
        if os.path.exists(os.path.join(self.root, f"labels/{split}")):
            self.labels_dir = os.path.join(self.root, f"labels/{split}")
        else:
            self.labels_dir = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")

        # Load labels
        labels = []
        if self.labels_dir:
            label_path = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_path)

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        class_id, x_center, y_center, width, height = map(
                            float, line.strip().split()
                        )
                        labels.append(
                            [int(class_id), x_center, y_center, width, height]
                        )

        # labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
        labels = np.array(labels) if labels else np.array([]).reshape(0, 5)

        # Resize image
        img = img.resize((self.img_size, self.img_size))

        return img, labels


def show_images_and_bboxes(dataset, images, labels_list, cols=4):
    # Ensure we are dealing with lists
    if not isinstance(images, (list, np.ndarray)):
        images = [images]
        labels_list = [labels_list]

    num_imgs = len(images)
    actual_cols = min(num_imgs, cols)
    rows = math.ceil(num_imgs / actual_cols)

    fig, axes = plt.subplots(rows, actual_cols, figsize=(4 * actual_cols, 4 * rows))

    # Flatten axes for easy iteration, handle single-image case
    if num_imgs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < num_imgs:
            img = images[i]
            labels = np.array(labels_list[i])

            # Convert PIL to numpy if necessary
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            ax.imshow(img)
            img_h, img_w = img.shape[:2]

            if labels.size > 0:
                for label in labels:
                    # Support for predicted labels which might have a confidence score [cls, x, y, w, h, conf]
                    # We only take the first 5 elements for logic
                    class_id, x_center, y_center, width, height = label[:5]

                    class_name = dataset.mapping.get(int(class_id), str(int(class_id)))

                    # Scale YOLO format (0-1) to pixel values
                    x_px = x_center * img_w
                    y_px = y_center * img_h
                    w_px = width * img_w
                    h_px = height * img_h

                    x1 = x_px - w_px / 2
                    y1 = y_px - h_px / 2

                    rect = patches.Rectangle(
                        (x1, y1),
                        w_px,
                        h_px,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x1,
                        y1 - 5,
                        class_name,
                        color="yellow",
                        fontsize=10,
                        weight="bold",
                        backgroundcolor="black",
                    )

            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
