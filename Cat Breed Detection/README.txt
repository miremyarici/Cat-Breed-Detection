# CAT BREED DETECTION

This project utilizes YOLOv4-Tiny to detect 12 different cat breeds in real-time via webcam. The model has been trained using the Oxford-IIIT Pet Dataset with OpenCV and Darknet libraries in Python. The training was done on Google Colab, utilizing a T4 GPU.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/miremyarici/Cat-Breed-Detection.git
    ```
2. Install required dependencies:
    ```bash
    pip install opencv-python numpy
    ```
3. Place the following files in your project directory:
    - yolov4-tiny.cfg
    - yolov4-tiny_best.weights
    - obj.names

## Usage

1. Run the script:
    ```bash
    python breeddet.py
    ```
2. The webcam will open, and the model will start detecting cat breeds in real-time. It will show the breed name and confidence percentage on the screen.
3. Press **'q'** to exit the webcam feed.

## Example Output:
When the model detects a cat breed, it will display something like: BIRMAN %87