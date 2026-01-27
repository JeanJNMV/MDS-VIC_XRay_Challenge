import os
import yaml
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class XRayDataset(Dataset):
    def __init__(self, yaml_file, split="train", resize=True, img_size=416):
        # Load yaml file
        with open(yaml_file, "r") as f:
            cfg = yaml.safe_load(f)

        # Dataset parameters
        yaml_dir = os.path.dirname(yaml_file)
        self.root = yaml_dir
        self.split = split
        self.resize = resize
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

        # Resize image if needed
        if self.resize:
            img = img.resize((self.img_size, self.img_size))

        # Convert image to tensor
        # img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        return img, labels


def show_image_and_labels(dataset, index):
    img, labels = dataset[index]

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    if labels.size > 0:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            class_name = dataset.mapping[int(class_id)]

            # YOLO format: x_center, y_center, width, height
            img_h, img_w = img.shape[:2]
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h

            # Convert to top-left corner
            x1 = x_center - width / 2
            y1 = y_center - height / 2

            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, class_name, color="yellow", fontsize=12, weight="bold")

    ax.axis("off")
    plt.show()
