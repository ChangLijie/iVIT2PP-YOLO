from typing import List, Optional

from pydantic import BaseModel, Field


class ImageFormat(BaseModel):
    """
    Class representing the format of an image in the dataset.

    Attributes:
        file_name (str): The name of the image file.
        height (int): The height of the image in pixels.
        width (int): The width of the image in pixels.
        id (int): The unique identifier of the image.
    """

    file_name: str
    height: int
    width: int
    id: int


class AnnotationFormat(BaseModel):
    """
    Class representing the format of an annotation in the dataset.

    Attributes:
        area (int): The area of the bounding box.
        iscrowd (int): Indicator if the annotation is a crowd (1) or not (0).
        image_id (int): The ID of the image associated with the annotation.
        bbox (List[int]): The bounding box coordinates [x1, y1, w, h].
        category_id (int): The category ID of the object.
        id (int): The unique identifier of the annotation.
        ignore (int): Indicator if the annotation should be ignored (1) or not (0).
        segmentation (List[Optional[List[int]]]): Segmentation data for the annotation.
    """

    area: int
    iscrowd: int
    image_id: int
    bbox: List[int]
    category_id: int
    id: int
    ignore: int
    segmentation: List[Optional[List[int]]]


class CategoryFormat(BaseModel):
    """
    Class representing the format of a category in the dataset.

    Attributes:
        supercategory (str): The supercategory of the object.
        id (int): The unique identifier of the category.
        name (str): The name of the category.
    """

    supercategory: str
    id: int
    name: str


class AnnotationsData(BaseModel):
    """
    Class representing the annotations data in the dataset.

    Attributes:
        images (list): A list of images in the dataset.
        type (Optional[str]): The type of the dataset (e.g., 'instances').
        annotations (list): A list of annotations in the dataset.
        categories (list): A list of categories in the dataset.
    """

    images: list
    type: Optional[str] = Field(default=None)
    annotations: list
    categories: list
