# iVIT-T to PP-YOLO

## Overview

This script is designed to convert datasets from the [iVIT-T](https://github.com/InnoIPA/ivit-t) format to the [PP-YOLO format](https://aistudio.baidu.com/datasetoverview). 

## Usage

### Command-Line Arguments

| Argument             | Required | Description                                                                                |
|----------------------|----------|--------------------------------------------------------------------------------------------|
| `-h, --help`         | No       | Show the help message and exit.                                                            |
| `-d, --dataset`      | Yes      | The path of the iVIT-T format dataset. Must be a valid path.                               |
| `-s, --save_dataset` | Yes      | The path where the converted dataset will be saved.                                        | 
| `-n, --limit_num`    | No       | How many numbers of the dataset you want to create. Default is None.                       |
| `-l, --limit_size`   | No       | The size of the image you want to process. Default is None.                                | 
| `-r, --rename`       | No       | Rename the images. Use as a boolean flag. Default is false.                                | 

### Running the Script

To run the script, use the following command:

```bash
python3 convert_dataset.py -d <dataset_path> -s <save_dataset_path> [-n <limit_num>] [-l <limit_size>] [-r]
```
<div align="center">
  <img width="100%" height="100%" src="docs/convert.png">
</div>

## Other
* [Todo](/docs/TODO.md)