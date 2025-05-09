# Rock Paper Scissors Game with Computer Vision

## CS402.3 Computer Graphics and Visualization - Coursework

This project implements a Rock Paper Scissors game that uses computer vision to detect hand gestures. The application captures the user's hand gesture through a camera, processes the image to recognize the gesture, and plays the game against the computer following standard Rock Paper Scissors rules.

## Features

- Real-time hand gesture recognition using computer vision
- Step-by-step visualization of image processing algorithms
- Classic mode (Rock, Paper, Scissors) and Extended mode (Rock, Paper, Scissors, Lizard, Spock)
- User-friendly graphical interface
- Detailed processing pipeline with multiple stages
- Background removal and hand segmentation
- Score tracking system

## Requirements

To run this application, you need:

- Python 3.7 or higher
- A webcam or camera connected to your computer

### Dependencies

The following Python libraries are required:

```
opencv-python
numpy
matplotlib
mediapipe
pillow
tkinter (typically included with Python)
```

You can install them using pip:

```
pip install -r requirements.txt
```

## Project Structure

The project consists of the following components:

- `rps_game.py`: Core game logic and basic image processing
- `image_processing_utils.py`: Advanced image processing utilities
- `rps_gui.py`: Graphical user interface
- `main.py`: Main entry point for the application
- `requirements.txt`: List of dependencies

## Running the Application

To run the application:

1. Install the dependencies as described above
2. Run the main script:

```
python main.py
```

By default, this launches the GUI version. To run the command-line interface version:

```
python main.py --mode cli
```

## How to Play

1. Launch the application
2. Click "Start Game" to begin
3. When prompted, say "Rock, Paper, Scissors, Shoot!" and show your hand gesture to the camera
4. The application will detect your gesture, choose a gesture for the computer, and determine the winner
5. The result will be displayed on the screen
6. Scores are updated automatically

## Game Rules

### Classic Mode

- Rock beats Scissors
- Scissors beats Paper
- Paper beats Rock
- Same gestures result in a tie

### Extended Mode

- Rock beats Scissors and Lizard
- Paper beats Rock and Spock
- Scissors beats Paper and Lizard
- Lizard beats Paper and Spock
- Spock beats Rock and Scissors
- Same gestures result in a tie

## Implementation Details

### Image Processing Pipeline

The application uses the following image processing steps:

1. **Image Capture**: Capturing frames from the camera
2. **Image Enhancement**: Adjusting brightness and contrast to improve visibility
3. **Background Removal**: Removing the background to isolate the hand
4. **Color Segmentation**: Detecting skin-colored regions to identify the hand
5. **Edge Detection**: Finding edges to outline the hand shape
6. **Contour Detection**: Extracting hand contours
7. **Feature Analysis**: Analyzing contours to identify fingers and hand shape
8. **Gesture Classification**: Classifying the hand gesture as rock, paper, or scissors

### Hand Gesture Recognition Techniques

The application uses several techniques for hand gesture recognition:

1. **Contour Analysis**: Detecting the outline of the hand
2. **Convexity Defects**: Identifying fingers by analyzing the spaces between them
3. **MediaPipe Hands**: Using Google's MediaPipe library for more accurate hand landmark detection
4. **Color-based Segmentation**: Isolating the hand using skin color detection

## License

This project is created for educational purposes as part of the CS402.3 Computer Graphics and Visualization coursework at NSBM Green University.

## Acknowledgements

- Dr. Rasika Ranaweera, Module Leader
- NSBM Green University
- OpenCV and MediaPipe documentation and communities
