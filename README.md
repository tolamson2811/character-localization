## HanNom Character Localization on Historical Document Images

This project employs the YOLOv5 model to detect and localize HanNom characters within scanned images of historical documents.

### Description

The project aims to digitize HanNom documents, facilitating indexing, annotation, and translation into modern Vietnamese. Leveraging deep learning with the YOLOv5 model, the project automates the HanNom character localization process, replacing the laborious manual approach.

### Data

* **Image Data:** 70 jpg images for training and 10 jpg images for testing.
* **Label Data:**  Corresponding .txt files for each image, containing information on the class, center coordinates (x_center, y_center), width, and height of each character.

The data was provided by Dr. Ta Viet Cuong - UET.

### Methodology

1. **Model Training:**
    * Utilized YOLOv5 on Google Colab with epochs = 300, batch size = 32.
    * The model was saved as `best.pt` (3.7MB).

2. **Character Localization:**
    * Loaded the `best.pt` model into the improved YOLOv5 source code.
    * Integrated a sample-scorer for performance evaluation.
    * Removed redundant code, keeping only necessary libraries.

### Results

* **Accuracy (mAP):** Achieved approximately 0.98 on both the training and testing datasets.
* **Runtime:** Averaged 8.93 seconds/image for the test set (10 images) and 9.2 seconds/image for the training set (70 images).

### Usage Instructions

1. Install the libraries listed in `requirements.txt`.
2. Run the following command, replacing the image and label directory paths accordingly:

    * For the validation dataset:
    ```
    python main.py .\data\images\val\ .\data\labels\val\ 
    ```

    * For the training dataset:
    ```
    python main.py .\data\images\train\ .\data\labels\train\
    ```

### Source Code

The source code is available at: [https://github.com/tolamson2811/character-localization](https://github.com/tolamson2811/character-localization)

### References

* [1] YOLOv5 Model Source Code: https://github.com/ultralytics/yolov5
* [2] Training YOLOv5 with Custom Data: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/
* [3] Accuracy Calculation Method: https://docs.ultralytics.com/guides/yolo-performance-metrics/#interpreting-the-output

### Notes

This project was developed as a final project for the Image Processing course.
