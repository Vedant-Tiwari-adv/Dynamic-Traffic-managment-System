Markdown

# 🚦 Adaptive Traffic Signal Control using YOLOv5

![Alt text for your image](DEMO%20trafic.png)

A smart traffic management system that leverages the YOLOv5 object detection model to analyze a video feed, count vehicles, and dynamically control a traffic signal based on real-time traffic density.

---

## 📸 Demo

![Demo GIF of the traffic system](https://user-images.githubusercontent.com/your-username/your-repo/demo.gif)

*(Replace the URL above with a link to a GIF or screenshot of your project in action. A visual demo is highly effective!)*

---

## ✨ Key Features

-   **🚗 Real-time Vehicle Detection:** Utilizes a pre-trained YOLOv5s model to accurately detect cars, buses, and trucks.
-   **🎯 Region of Interest (ROI):** Focuses detection on a specific, user-defined area of the frame for improved performance and accuracy.
-   **💡 Dynamic Signal Logic:** Implements a heuristic-based algorithm to switch the traffic light to green when traffic volume exceeds a threshold.
-   **🖥️ Rich Visual Feedback:** Overlays bounding boxes, detection confidence, the ROI, vehicle count, and the current signal status (🔴/🟢) on the output video.
-   **⌨️ Interactive Controls:** Allows pausing/resuming the video feed with the **spacebar** and quitting with the **'q'** key.

---

## 🛠️ How It Works

The application follows a straightforward pipeline:

1.  **Video Input:** Reads frames from a local video file or a live webcam feed.
2.  **ROI Extraction:** Isolates a specific rectangular region of the frame where traffic is to be monitored.
3.  **Object Detection:** The extracted ROI is passed to the YOLOv5 model, which returns the bounding boxes and classes of all detected vehicles.
4.  **Counting & Filtering:** The script processes the detections, filters out small or low-confidence objects, and counts the remaining vehicles. It also subtracts a predefined number of `PARKED_CARS` to handle static obstacles.
5.  **Signal Decision:** The core logic determines the state of the traffic signal. If the number of waiting cars is above a set threshold, the signal turns green.
6.  **Visualization:** The final frame, complete with all visual overlays, is rendered and displayed on the screen.

---

## 🚀 Getting Started

Follow these steps to get the project set up and running on your local machine.

### 1. Prerequisites

-   Python 3.8 or newer
-   `pip` package manager

### 2. Installation

First, clone the repository to your local machine:
bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

Next, it's highly recommended to create and activate a virtual environment:
Bash

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Finally, install the required packages:
Bash

pip install -r requirements.txt

(Note: For this to work, create a requirements.txt file and add the following lines to it:)
Plaintext

torch
opencv-python
numpy
ultralytics

3. Running the Application

Ensure you have a video file named traffic2.mp4 in the project's root directory. Then, run the main script:
Bash

python your_script_name.py

A window should appear, showing the processed video feed.

🔧 Configuration

You can easily customize the application's behavior by modifying the variables at the top of the script.
Variable	Type	Description	Default Value
model.classes	list	A list of COCO class IDs to detect.	[2, 5, 7]
region_top_left	tuple	The (x1, y1) coordinates for the top-left corner of the ROI.	(220, 2)
region_bottom_right	tuple	The (x2, y2) coordinates for the bottom-right corner of the ROI.	(560, 170)
PARKED_CARS	int	A fixed number of cars to subtract from the count, useful for ignoring permanently parked vehicles.	6
VIDEO_SOURCE	string	The path to the input video file. Change to 0 to use the primary webcam.	"traffic2.mp4"
MIN_AREA	int	The minimum pixel area for a detected bounding box to be considered a valid vehicle.	400

💻 Technologies Used

    PyTorch: For running the deep learning model.

    YOLOv5 (from Ultralytics): For state-of-the-art object detection.

    OpenCV (cv2): For all video processing, drawing, and display tasks.

    NumPy: For efficient numerical operations and array manipulation.

📄 License

This project is licensed under the MIT License - see the LICENSE.md file for details.
