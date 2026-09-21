import os

from pycocotools.coco import COCO


def coco_to_yolo(coco_json, image_dir, label_dir, class_names):
    # 创建标签输出目录
    os.makedirs(label_dir, exist_ok=True)

    coco = COCO(coco_json)
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

        label_path = os.path.join(label_dir, file_name.rsplit(".", 1)[0] + ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        with open(label_path, "w") as f:
            for ann in annotations:
                category_id = ann["category_id"]
                bbox = ann["bbox"]

                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height

                class_id = category_id - 1  # COCO是从1开始，YOLO从0开始
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        print(f"[✓] 转换完成：{file_name}")


# FLIR 数据集类名顺序（按 coco.json 中顺序）
class_names = [
    "person",
    "bike",
    "car",
    "motor",
    "bus",
    "train",
    "truck",
    "light",
    "hydrant",
    "sign",
    "dog",
    "deer",
    "skateboard",
    "stroller",
    "scooter",
    "other vehicle",
]

# 设置路径（根据你实际下载的位置）
train_json = "C:/Users/blackvccat/Downloads/FLIR_ADAS_v2/images_thermal_train/coco.json"
train_images = "C:/Users/blackvccat/Downloads/FLIR_ADAS_v2/images_thermal_train"
train_labels = "C:/Users/blackvccat/PycharmProjects/PythonProject/ultralytics/datasets/labels/train"

val_json = "C:/Users/blackvccat/Downloads/FLIR_ADAS_v2/images_thermal_val/coco.json"
val_images = "C:/Users/blackvccat/Downloads/FLIR_ADAS_v2/images_thermal_val"
val_labels = "C:/Users/blackvccat/PycharmProjects/PythonProject/ultralytics/datasets/labels/val"

# 执行转换
print("=== 开始转换训练集 ===")
coco_to_yolo(train_json, train_images, train_labels, class_names)

print("=== 开始转换验证集 ===")
coco_to_yolo(val_json, val_images, val_labels, class_names)
