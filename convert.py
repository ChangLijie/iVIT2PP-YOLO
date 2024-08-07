# ivit-t dataset format is [index,x1,x2,w,h]
# PPYolo is coco dataset , and the bbox format is [x1,x2,w,h]

import argparse
import json
import os
import shutil
from argparse import SUPPRESS, ArgumentParser
from os import walk

import cv2
from tqdm import tqdm

from tools.logger import config_logger
from tools.ppyolo_format import (
    AnnotationFormat,
    AnnotationsData,
    CategoryFormat,
    ImageFormat,
)


def build_argparser():
    """
    Build and return the argument parser.

    This function creates and configures an ArgumentParser for the script, including
    various command-line arguments that the user can provide.

    Returns:
        ArgumentParser: Configured ArgumentParser object.
    """

    def valid_path(path):
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"Invalid path: {path}")
        return path

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument(
        "-h",
        "--help",
        action="help",
        default=SUPPRESS,
        help="Show this help message and exit.",
    )
    args.add_argument(
        "-n",
        "--limit_num",
        default=None,
        help="How many num of dataset you want to create!",
    )
    args.add_argument(
        "-d",
        "--dataset",
        type=valid_path,
        help="The path of iVIT-T format dataset. Must be a valid path.",
    )
    args.add_argument(
        "-l",
        "--limit_size",
        default=None,
        help="The size of image you want to pass",
    )
    args.add_argument(
        "-s",
        "--save_dataset",
        required=True,
        help="The path of dataset after convert!",
    )
    args.add_argument("-r", "--rename", action="store_true", help="Rename image!")

    return parser


def relieve_nor(img_path: str, x1: float, y1: float, w: float, h: float) -> tuple:
    """
    Normalize and calculate the bounding box coordinates in the original image.

    Args:
        img_path (str): Path to the image.
        x1 (float): Normalized x1 coordinate.
        y1 (float): Normalized y1 coordinate.
        w (float): Normalized width of the bounding box.
        h (float): Normalized height of the bounding box.

    Returns:
        tuple: A tuple containing the image dimensions (width, height), and the calculated (x1, y1, w, h) coordinates.
    """
    image = cv2.imread(img_path)

    w = int(w * image.shape[1])
    h = int(h * image.shape[0])

    x1 = int(x1 * (image.shape[1]) - (w // 2))
    y1 = int(y1 * (image.shape[0]) - (h // 2))

    return (image.shape[1], image.shape[0]), x1, y1, w, h


def read_annotation(annotation_path: str) -> list:
    """
    Read and parse the annotation file.

    Args:
        annotation_path (str): Path to the annotation file.

    Returns:
        list: A list of annotations where each annotation is a list of [class_id, x1, y1, w, h].
    """
    f = open(annotation_path, "r")
    result = []
    for line in f.readlines():
        result.append(
            [
                int(line.split(" ")[0]),
                float(line.split(" ")[1]),
                float(line.split(" ")[2]),
                float(line.split(" ")[3]),
                float(line.split(" ")[4]),
            ]
        )
    f.close()
    return result


def calculate_area(x1: int, y1: int, w: int, h: int) -> int:
    """
    Calculate the area of a bounding box.

    Args:
        x1 (int): x-coordinate of the top-left corner.
        y1 (int): y-coordinate of the top-left corner.
        w (int): Width of the bounding box.
        h (int): Height of the bounding box.

    Returns:
        int: The area of the bounding box.
    """
    area = w * h

    return area


def save_json(data: AnnotationsData, save_path: str, train: bool = True) -> None:
    """
    Save the dataset annotations in JSON format.

    Args:
        data (AnnotationsData): Annotations data to be saved.
        s_path (str): Path to save the JSON file.
        train (bool, optional): Flag to indicate if the data is for training. Defaults to True.
    """
    data_dict = data.dict()
    if train:
        s_file = "annotations/instance_train.json"
    else:
        s_file = "annotations/instance_val.json"
    save_dst = os.path.join(save_path, s_file)
    with open(save_dst, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)


def create_ppyolo_folder(save_path: str) -> None:
    """
    Create the necessary folder structure for the PP-YOLO dataset.

    This function creates the 'annotations' and 'images' folders in the specified
    save path if they do not already exist. These folders are required for the
    PP-YOLO dataset format.

    Args:
        save_path (str): The base path where the folders will be created.

    Returns:
        None
    """
    anno_folder_name = "annotations"
    image_folder_name = "images"
    anno_folder = os.path.join(save_path, anno_folder_name)
    image_folder = os.path.join(save_path, image_folder_name)
    if not os.path.exists(anno_folder):
        os.makedirs(anno_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)


if __name__ == "__main__":
    args = build_argparser().parse_args()
    logger = config_logger(
        log_name="system.log",
        logger_name="system",
        default_folder="./log",
        write_mode="w",
        level="debug",
    )
    logger.info("--------------User setting--------------")
    mypath = args.dataset
    logger.info(f"Dataset : '{mypath}'.")
    save_path = args.save_dataset
    logger.info(f"Save : '{save_path}'.")
    limit_num = args.limit_num
    logger.info(f"Limit number : '{limit_num}'.")
    limit_size = args.limit_size
    logger.info(f"Limit file size : '{limit_size}'.")
    rename = args.rename
    logger.info(f"Rename : '{rename}'.")
    SUPPORT_EXTENSION = [".png", ".jpeg", ".jpg"]
    logger.info(f"Support image extensions : '{SUPPORT_EXTENSION}'.")

    logger.info("--------------Start convert--------------")
    create_ppyolo_folder(save_path)
    logger.info(f"Create ppyolo folder : '{save_path}'.")

    images = []
    annotations = []
    categories = []
    img_id = 1
    annotations_id = 1
    fail_file = []
    over_size = []
    no_annotation = []
    for root, dirs, files in walk(mypath):
        for file in tqdm(files, desc="Processing files"):
            if limit_num:
                if img_id == int(limit_num):
                    logger.warning(f"Stop convert. (Deal image number: '{img_id}'.)")

            file_name = os.path.basename(file)
            base_name, ext = os.path.splitext(file_name)
            ext = ext.lower()
            # file_split = file.split('.')
            if ext == ".txt":
                class_txt_path = os.path.join(root, file)
                f = open(class_txt_path, "r")
                for idx, label in enumerate(f.readlines()):
                    categories.append(
                        CategoryFormat(
                            supercategory="none", id=idx, name=label.strip("\n")
                        )
                    )

            elif ext in SUPPORT_EXTENSION:
                img_name = base_name + ext
                annotation = base_name + ".txt"

                anno_path = os.path.join(root, annotation)
                if not os.path.exists(anno_path):
                    no_annotation.append(file)
                    continue
                img_path = os.path.join(root, img_name)

                if limit_size:
                    if os.path.getsize(img_path) > int(limit_size):
                        over_size.append(file)
                        continue
                if rename:
                    re_img_name = str(img_id) + ext
                    new_img_path = os.path.join(save_path, "images", re_img_name)

                else:
                    new_img_path = os.path.join(save_path, "images", img_name)
                shutil.copy(img_path, new_img_path)

                bboxes = read_annotation(anno_path)
                image = cv2.imread(new_img_path)

                for bbox in bboxes:
                    cls_id, raw_x1, raw_y1, raw_w, raw_h = bbox

                    img_shape, x1, y1, w, h = relieve_nor(
                        new_img_path, raw_x1, raw_y1, raw_w, raw_h
                    )

                    area = calculate_area(x1, y1, w, h)
                    annotations.append(
                        AnnotationFormat(
                            area=area,
                            iscrowd=0,
                            image_id=img_id,
                            bbox=[x1, y1, w, h],
                            category_id=cls_id,
                            id=annotations_id,
                            ignore=0,
                            segmentation=[],
                        )
                    )
                    annotations_id += 1
                #     draw_0 = cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0,255,0), 2)
                # cv2.imwrite("./img_{}.jpg".format(str(img_id)),draw_0)
                if rename:
                    images.append(
                        ImageFormat(
                            file_name=re_img_name,
                            height=img_shape[1],
                            width=img_shape[0],
                            id=img_id,
                        )
                    )
                else:
                    images.append(
                        ImageFormat(
                            file_name=img_name,
                            height=img_shape[1],
                            width=img_shape[0],
                            id=img_id,
                        )
                    )
                img_id += 1
            else:
                fail_file.append(file)

    # print(categories,'\n')
    # print(annotations)
    # print(images)
    train_format = AnnotationsData(
        images=images, type="instances", annotations=annotations, categories=categories
    )

    eval_format = AnnotationsData(images=[], annotations=[], categories=categories)
    logger.info("--------------Failed convert file--------------")
    logger.warning(f"Not support extension :{fail_file}")
    logger.warning(f"Filed to find annotation :{no_annotation}")
    logger.warning(f"Over size :{over_size}")
    save_json(train_format, save_path)
    logger.info("Success covert instance_train.json!")
    save_json(eval_format, save_path, train=False)
    logger.info("Success covert instance_val.json!")

    logger.info("Finish covert!")
