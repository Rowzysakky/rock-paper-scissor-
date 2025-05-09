"""
Rock Paper Scissors Game with Computer Vision
CS402.3 Coursework

This prototype demonstrates the basic functionality of a Rock Paper Scissors game
that uses computer vision to detect hand gestures.

Key components:
1. Camera capture module
2. Image processing pipeline
3. Hand gesture recognition
4. Game logic
5. Visualization

Required libraries:
- OpenCV (cv2) for image processing and camera capture
- NumPy for numerical operations
- Matplotlib for visualization of processing steps
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from enum import Enum


class Gesture(Enum):
    """Enum for the different hand gestures"""
    ROCK = 1
    PAPER = 2
    SCISSORS = 3
    UNKNOWN = 4


class GameResult(Enum):
    """Enum for the possible game results"""
    WIN = 1
    LOSE = 2
    TIE = 3


class RockPaperScissors:
    """
    Main class for the Rock Paper Scissors game
    """
    
    def __init__(self):
        """Initialize the game"""
        self.cap = None
        self.processing_images = []
        self.user_gesture = None
        self.computer_gesture = None
        self.result = None
    
    def initialize_camera(self):
        """Initialize the camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        return True
    
    def release_camera(self):
        """Release the camera capture"""
        if self.cap is not None:
            self.cap.release()
    
    def capture_image(self):
        """Capture an image from the camera"""
        if self.cap is None:
            print("Error: Camera not initialized.")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return None
        
        # Store original image for visualization
        self.processing_images = [("Original", frame.copy())]
        
        return frame
    
    def process_image(self, image):
        """
        Process the image to detect hand gesture
        Steps:
        1. Convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Apply thresholding to create binary image
        4. Find contours to detect hand shape
        """
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.processing_images.append(("Grayscale", gray.copy()))
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        self.processing_images.append(("Blurred", blurred.copy()))
        
        # Apply thresholding
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.processing_images.append(("Thresholded", thresh.copy()))
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on a copy of the original image
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        self.processing_images.append(("Contours", contour_image.copy()))
        
        return contours
    
    def recognize_gesture(self, contours, image):
        """
        Recognize the hand gesture based on contours
        This is a simplified version that needs to be expanded
        """
        if not contours:
            return Gesture.UNKNOWN
        
        # Find the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Calculate convex hull
        hull = cv2.convexHull(max_contour)
        
        # Find convexity defects
        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        if len(max_contour) > 3 and len(hull_indices) > 3:
            defects = cv2.convexityDefects(max_contour, hull_indices)
        else:
            return Gesture.ROCK  # Default to rock if not enough points
        
        # Count fingers based on defects
        finger_count = 0
        
        # Draw defects and count fingers
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                
                # Calculate angle between vectors
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                # Apply cosine law
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                
                # Fingers are counted as angles less than 90 degrees
                if angle <= np.pi / 2:  # 90 degrees
                    finger_count += 1
                    # Draw circles at the defect points
                    cv2.circle(image, far, 5, [0, 0, 255], -1)
        
        # Create a copy with gesture visualization
        gesture_image = image.copy()
        cv2.putText(gesture_image, f"Fingers: {finger_count}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        self.processing_images.append(("Gesture Detection", gesture_image.copy()))
        
        # Simple finger count to gesture mapping
        # This is a simplified version and needs refinement
        if finger_count <= 1:
            return Gesture.ROCK
        elif finger_count == 2:
            return Gesture.SCISSORS
        else:
            return Gesture.PAPER
    
    def get_computer_gesture(self):
        """Generate a random gesture for the computer"""
        return random.choice([Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS])
    
    def determine_winner(self, user_gesture, computer_gesture):
        """Determine the winner based on the gestures"""
        if user_gesture == computer_gesture:
            return GameResult.TIE
        
        if user_gesture == Gesture.ROCK:
            return GameResult.WIN if computer_gesture == Gesture.SCISSORS else GameResult.LOSE
        
        if user_gesture == Gesture.PAPER:
            return GameResult.WIN if computer_gesture == Gesture.ROCK else GameResult.LOSE
        
        if user_gesture == Gesture.SCISSORS:
            return GameResult.WIN if computer_gesture == Gesture.PAPER else GameResult.LOSE
        
        return None
    
    def display_processing_steps(self):
        """Display all the image processing steps"""
        num_images = len(self.processing_images)
        if num_images == 0:
            return
        
        cols = 2
        rows = (num_images + cols - 1) // cols
        
        plt.figure(figsize=(15, 10))
        
        for i, (title, image) in enumerate(self.processing_images):
            plt.subplot(rows, cols, i + 1)
            
            # Handle grayscale images
            if len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def display_result(self, image):
        """Display the game result"""
        result_image = image.copy()
        
        # Draw user and computer gestures
        user_text = f"You: {self.user_gesture.name}"
        comp_text = f"Computer: {self.computer_gesture.name}"
        
        cv2.putText(result_image, user_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_image, comp_text, (10, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw result
        if self.result == GameResult.WIN:
            result_text = "Result: You Win!"
            color = (0, 255, 0)  # Green
        elif self.result == GameResult.LOSE:
            result_text = "Result: You Lose!"
            color = (0, 0, 255)  # Red
        else:
            result_text = "Result: Tie!"
            color = (255, 255, 0)  # Yellow
        
        cv2.putText(result_image, result_text, (10, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display image
        cv2.imshow("Game Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def play_game(self):
        """Main game loop"""
        print("Starting Rock Paper Scissors game...")
        print("Say 'Rock, Paper, Scissors, Shoot!' and show your gesture")
        
        if not self.initialize_camera():
            return
        
        # Wait for user to get ready
        print("Get ready to show your gesture...")
        time.sleep(3)
        print("Capturing now!")
        
        # Capture image
        image = self.capture_image()
        if image is None:
            self.release_camera()
            return
        
        # Process image
        contours = self.process_image(image)
        
        # Recognize gesture
        self.user_gesture = self.recognize_gesture(contours, image)
        print(f"Detected gesture: {self.user_gesture.name}")
        
        # Get computer gesture
        self.computer_gesture = self.get_computer_gesture()
        print(f"Computer chose: {self.computer_gesture.name}")
        
        # Determine winner
        self.result = self.determine_winner(self.user_gesture, self.computer_gesture)
        
        if self.result == GameResult.WIN:
            print("You win!")
        elif self.result == GameResult.LOSE:
            print("You lose!")
        else:
            print("It's a tie!")
        
        # Display processing steps
        self.display_processing_steps()
        
        # Display final result
        self.display_result(image)
        
        # Release camera
        self.release_camera()


def main():
    """Main function"""
    game = RockPaperScissors()
    game.play_game()


if __name__ == "__main__":
    main()