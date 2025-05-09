"""
Rock Paper Scissors Game - Main Script
CS402.3 Coursework

This is the main entry point for the Rock Paper Scissors game application.
It integrates all components of the game and provides a complete playing experience.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from enum import Enum


# Import game modules
try:
    from rps_game import RockPaperScissors
    from image_processing_utils import (
        HandGestureRecognizer, 
        BackgroundRemover, 
        ImageEnhancer,
        ColorSegmenter,
        create_processing_montage
    )
    from rps_gui import RockPaperScissorsGUI
except ImportError:
    # For development and testing, dynamically import from current directory
    import importlib.util
    
    def import_module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Import modules dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    rps_game = import_module_from_file(
        "rps_game", 
        os.path.join(current_dir, "rps_game.py")
    )
    RockPaperScissors = rps_game.RockPaperScissors
    
    image_processing = import_module_from_file(
        "image_processing_utils", 
        os.path.join(current_dir, "image_processing_utils.py")
    )
    HandGestureRecognizer = image_processing.HandGestureRecognizer
    BackgroundRemover = image_processing.BackgroundRemover
    ImageEnhancer = image_processing.ImageEnhancer
    ColorSegmenter = image_processing.ColorSegmenter
    create_processing_montage = image_processing.create_processing_montage
    
    rps_gui = import_module_from_file(
        "rps_gui", 
        os.path.join(current_dir, "rps_gui.py")
    )
    RockPaperScissorsGUI = rps_gui.RockPaperScissorsGUI


class RockPaperScissorsApp:
    """
    Main application class for Rock Paper Scissors game
    """
    
    def __init__(self, mode="gui"):
        """
        Initialize the application
        Args:
            mode (str): Application mode - "gui" for graphical interface, "cli" for command line
        """
        self.mode = mode
        
        # Initialize game components
        self.game = RockPaperScissors()
        self.hand_recognizer = HandGestureRecognizer()
        self.bg_remover = BackgroundRemover()
        self.image_enhancer = ImageEnhancer()
        self.color_segmenter = ColorSegmenter()
    
    def run(self):
        """Run the application"""
        if self.mode == "gui":
            # Run GUI version
            import tkinter as tk
            root = tk.Tk()
            app = RockPaperScissorsGUI(root, "Rock Paper Scissors Game")
        else:
            # Run CLI version
            self.run_cli()
    
    def run_cli(self):
        """Run command line interface version"""
        print("=== Rock Paper Scissors Game ===")
        print("Command Line Interface Version")
        print("Press 'q' to quit at any time")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        try:
            while True:
                # Show camera feed
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Show frame
                cv2.imshow("Rock Paper Scissors", frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.play_round(frame)
        
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
    
    def play_round(self, frame):
        """
        Play a round of Rock Paper Scissors
        Args:
            frame (numpy.ndarray): Captured frame to process
        """
        # Process image
        print("Processing image...")
        
        # Step 1: Enhance image
        enhanced = self.image_enhancer.adjust_brightness_contrast(frame)
        
        # Step 2: Remove background
        _, segmented = self.bg_remover.remove_background_static(enhanced)
        
        # Step 3: Extract skin
        skin_mask, skin = self.color_segmenter.extract_skin(segmented)
        
        # Step 4: Detect hand landmarks
        landmarks, annotated = self.hand_recognizer.detect_landmarks(frame)
        
        # Step 5: Recognize gesture
        if landmarks:
            user_gesture = self.hand_recognizer.recognize_gesture(landmarks)
            print(f"Detected gesture: {user_gesture.name}")
        else:
            from rps_game import Gesture
            user_gesture = Gesture.UNKNOWN
            print("No hand detected")
        
        # Get computer gesture
        computer_gesture = self.game.get_computer_gesture()
        print(f"Computer chose: {computer_gesture.name}")
        
        # Determine winner
        result = self.game.determine_winner(user_gesture, computer_gesture)
        
        if result:
            print(f"Result: {result.name}")
        else:
            print("No valid result")
        
        # Show processing steps
        processing_steps = [
            ("Original", frame),
            ("Enhanced", enhanced),
            ("Segmented", segmented),
            ("Skin Extraction", skin),
            ("Hand Detection", annotated)
        ]
        
        # Create montage
        montage = create_processing_montage(
            [img for _, img in processing_steps],
            [title for title, _ in processing_steps]
        )
        
        # Show results
        cv2.imshow("Processing Steps", montage)
        
        # Create result image
        result_img = frame.copy()
        
        # Add text for user and computer gestures
        cv2.putText(
            result_img,
            f"You: {user_gesture.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            result_img,
            f"Computer: {computer_gesture.name}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # Add result text
        if result:
            result_text = f"Result: {result.name}"
            color = (0, 255, 0) if result.name == "WIN" else (0, 0, 255) if result.name == "LOSE" else (255, 255, 0)
            
            cv2.putText(
                result_img,
                result_text,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )
        
        # Show result
        cv2.imshow("Game Result", result_img)
        cv2.waitKey(3000)  # Wait for 3 seconds


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Rock Paper Scissors Game')
    parser.add_argument(
        '--mode',
        choices=['gui', 'cli'],
        default='gui',
        help='Application mode: gui (graphical) or cli (command line)'
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    app = RockPaperScissorsApp(mode=args.mode)
    app.run()


if __name__ == "__main__":
    main()