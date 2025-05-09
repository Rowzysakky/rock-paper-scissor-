# Project Name

Rock Paper Scissors Game using Hand Gesture Recognition and Image Processing

## Introduction

This project presents a computer vision-based Rock Paper Scissors (RPS) game, allowing users to play the classic hand gesture game against a computer by recognizing hand gestures captured through a webcam. The system leverages OpenCV, MediaPipe, and custom image processing techniques for gesture detection and game control.

### Objectives

Develop a real-time hand gesture recognition system for Rock, Paper, and Scissors gestures.

Integrate image processing techniques to enhance image quality and improve detection accuracy.

Provide both Graphical User Interface (GUI) and Command Line Interface (CLI) modes for gameplay.

Implement background removal, skin segmentation, and landmark-based gesture classification.


### Technologies Used


| Technology | Purpose                    |
| :--------- | :------------------------- |
| Python 3.x | Programming Language       |
| OpenCV     | Image Capture & Processing |
| MediaPipe  | Hand Landmark Detection    |
| NumPy      | Numerical Computations     |
| Tkinter    | GUI Development            |


### System Architecture

image_processing_utils.py
Implements background removal, image enhancement, skin color segmentation, and hand gesture recognition using MediaPipe.

main.py
Main controller script handling gameplay, image capture, processing steps, and user interaction via GUI or CLI.

rps_game.py
(Assumed to contain game logic and gesture definitions.)

rps_gui.py
(Assumed to provide a GUI-based gameplay window.)



### Implementation Details


Hand Gesture Recognition
Utilizes MediaPipe's hand landmarks to detect and classify gestures based on the relative positions of finger tips and palm landmarks.

Recognized gestures:

Rock: No fingers extended.

Paper: All fingers extended.

Scissors: Index and middle fingers extended.



### Game Modes


CLI Mode:

Users interact through keypresses (e.g., q to quit, p to play) with real-time webcam capture.

GUI Mode:

A GUI window provides visual feedback and results (implementation assumed in rps_gui.py).



### Results


Real-time hand gesture detection through a webcam.

Accurate classification of Rock, Paper, and Scissors gestures.

Clean, visually annotated processing steps and results.

CLI and GUI modes offering flexible interaction.




### Conclusion

This project successfully integrates computer vision and image processing techniques for an interactive Rock Paper Scissors game using hand gestures. By leveraging MediaPipe and OpenCV, the system achieves real-time gesture recognition with effective preprocessing to enhance reliability.






## Group Member Contributions



| Name      | Contribution                                                                                 |
| :-------- | :--------------------------------------------------------------------------------------      |
| MRR.Sakky | Main application design, CLI implementation, image preprocessing, MediaPipe integration      |
| MRR.sakky | GUI development, result display management, testing                                          |
| Dilaxsana | Testing, debugging, support in image processing utilities, documentation preparation         |
| MM. Siraj | Documentation, code review, assistance in application testing                                |


