"""
Rock Paper Scissors Game GUI
CS402.3 Coursework

This module provides a graphical user interface for the Rock Paper Scissors game.
It uses Tkinter for creating the interface and displays camera feed, processing steps,
and game results.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import threading
import time
from enum import Enum
import sys
import os

# Import your modules here
# from rps_game import RockPaperScissors, Gesture, GameResult
# from image_processing import HandGestureRecognizer, BackgroundRemover, ImageEnhancer


class RockPaperScissorsGUI:
    """
    GUI for the Rock Paper Scissors game
    """
    
    def __init__(self, window, window_title):
        """Initialize the GUI"""
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#f0f0f0")
        
        # Set window size and position
        window_width = 1200
        window_height = 800
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Initialize game state
        self.game_running = False
        self.countdown_active = False
        self.processing_active = False
        
        # Create UI elements
        self.create_ui()
        
        # Initialize camera
        self.cap = None
        self.after_id = None
        
        # Close the window properly
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start UI update loop
        self.window.mainloop()
    
    def create_ui(self):
        """Create UI elements"""
        # Create main frames
        self.create_header_frame()
        self.create_main_frame()
        self.create_footer_frame()
    
    def create_header_frame(self):
        """Create the header frame with title and instructions"""
        header_frame = ttk.Frame(self.window, padding="10")
        header_frame.pack(fill=tk.X)
        
        # Title label
        title_label = ttk.Label(
            header_frame, 
            text="Rock Paper Scissors Game",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=10)
        
        # Instructions label
        instructions = (
            "1. Click 'Start Game' to begin\n"
            "2. Say 'Rock, Paper, Scissors, Shoot!' and show your gesture\n"
            "3. The computer will respond with its gesture\n"
            "4. The winner will be determined"
        )
        instructions_label = ttk.Label(
            header_frame, 
            text=instructions,
            font=("Arial", 12)
        )
        instructions_label.pack(pady=5)
    
    def create_main_frame(self):
        """Create the main frame with camera feed and processing steps"""
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for camera feed
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Camera Feed", padding="10")
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(self.camera_frame, bg="black")
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Countdown label
        self.countdown_label = ttk.Label(
            self.camera_canvas, 
            text="",
            font=("Arial", 72, "bold"),
            foreground="white",
            background="black"
        )
        self.countdown_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Right frame for processing steps
        self.processing_frame = ttk.LabelFrame(self.main_frame, text="Processing Steps", padding="10")
        self.processing_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create a canvas for processing steps
        self.processing_canvas = tk.Canvas(self.processing_frame, bg="white")
        self.processing_canvas.pack(fill=tk.BOTH, expand=True)
    
    def create_footer_frame(self):
        """Create the footer frame with game controls and results"""
        footer_frame = ttk.Frame(self.window, padding="10")
        footer_frame.pack(fill=tk.X)
        
        # Game controls
        controls_frame = ttk.Frame(footer_frame)
        controls_frame.pack(pady=10)
        
        # Start game button
        self.start_button = ttk.Button(
            controls_frame, 
            text="Start Game",
            command=self.start_game
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop game button
        self.stop_button = ttk.Button(
            controls_frame, 
            text="Stop Game",
            command=self.stop_game,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Game mode selector
        mode_frame = ttk.LabelFrame(footer_frame, text="Game Mode", padding="5")
        mode_frame.pack(pady=5)
        
        self.game_mode = tk.StringVar(value="classic")
        
        # Classic mode (Rock, Paper, Scissors)
        classic_radio = ttk.Radiobutton(
            mode_frame,
            text="Classic Mode (Rock, Paper, Scissors)",
            variable=self.game_mode,
            value="classic"
        )
        classic_radio.pack(anchor=tk.W, padx=5)
        
        # Extended mode (Rock, Paper, Scissors, Lizard, Spock)
        extended_radio = ttk.Radiobutton(
            mode_frame,
            text="Extended Mode (Rock, Paper, Scissors, Lizard, Spock)",
            variable=self.game_mode,
            value="extended"
        )
        extended_radio.pack(anchor=tk.W, padx=5)
        
        # Game results frame
        results_frame = ttk.LabelFrame(footer_frame, text="Game Results", padding="5")
        results_frame.pack(fill=tk.X, pady=5)
        
        # User gesture
        user_frame = ttk.Frame(results_frame)
        user_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=10)
        
        ttk.Label(user_frame, text="Your Gesture:", font=("Arial", 12, "bold")).pack(anchor=tk.CENTER)
        self.user_gesture_label = ttk.Label(user_frame, text="-", font=("Arial", 24))
        self.user_gesture_label.pack(anchor=tk.CENTER, pady=5)
        
        # Result
        result_frame = ttk.Frame(results_frame)
        result_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=10)
        
        ttk.Label(result_frame, text="Result:", font=("Arial", 12, "bold")).pack(anchor=tk.CENTER)
        self.result_label = ttk.Label(result_frame, text="-", font=("Arial", 24))
        self.result_label.pack(anchor=tk.CENTER, pady=5)
        
        # Computer gesture
        computer_frame = ttk.Frame(results_frame)
        computer_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=10)
        
        ttk.Label(computer_frame, text="Computer Gesture:", font=("Arial", 12, "bold")).pack(anchor=tk.CENTER)
        self.computer_gesture_label = ttk.Label(computer_frame, text="-", font=("Arial", 24))
        self.computer_gesture_label.pack(anchor=tk.CENTER, pady=5)
        
        # Score board
        score_frame = ttk.LabelFrame(footer_frame, text="Score", padding="5")
        score_frame.pack(fill=tk.X, pady=5)
        
        # User score
        user_score_frame = ttk.Frame(score_frame)
        user_score_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        ttk.Label(user_score_frame, text="You:", font=("Arial", 12)).pack(anchor=tk.CENTER)
        self.user_score_label = ttk.Label(user_score_frame, text="0", font=("Arial", 18, "bold"))
        self.user_score_label.pack(anchor=tk.CENTER)
        
        # Computer score
        computer_score_frame = ttk.Frame(score_frame)
        computer_score_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=True)
        
        ttk.Label(computer_score_frame, text="Computer:", font=("Arial", 12)).pack(anchor=tk.CENTER)
        self.computer_score_label = ttk.Label(computer_score_frame, text="0", font=("Arial", 18, "bold"))
        self.computer_score_label.pack(anchor=tk.CENTER)
    
    def start_game(self):
        """Start the game"""
        if not self.game_running:
            self.game_running = True
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            
            # Reset labels
            self.user_gesture_label.configure(text="-")
            self.computer_gesture_label.configure(text="-")
            self.result_label.configure(text="-")
            
            # Initialize scores if needed
            if not hasattr(self, 'user_score'):
                self.user_score = 0
                self.computer_score = 0
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.show_error("Could not open camera")
                self.stop_game()
                return
            
            # Start camera update
            self.update_camera()
            
            # Start countdown
            self.start_countdown()
    
    def stop_game(self):
        """Stop the game"""
        self.game_running = False
        self.countdown_active = False
        self.processing_active = False
        
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Cancel scheduled updates
        if self.after_id is not None:
            self.window.after_cancel(self.after_id)
            self.after_id = None
        
        # Clear canvases
        self.camera_canvas.delete("all")
        self.processing_canvas.delete("all")
        
        # Reset labels
        self.countdown_label.configure(text="")
    
    def update_camera(self):
        """Update the camera feed"""
        if self.cap is not None and self.game_running:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to display on canvas
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction
                
                # Resize frame to fit canvas
                canvas_width = self.camera_canvas.winfo_width()
                canvas_height = self.camera_canvas.winfo_height()
                
                if canvas_width > 0 and canvas_height > 0:
                    # Calculate aspect ratio
                    frame_height, frame_width = frame.shape[:2]
                    aspect_ratio = frame_width / frame_height
                    
                    # Resize to fit canvas while maintaining aspect ratio
                    if canvas_width / canvas_height > aspect_ratio:
                        # Canvas is wider than frame
                        display_height = canvas_height
                        display_width = int(display_height * aspect_ratio)
                    else:
                        # Canvas is taller than frame
                        display_width = canvas_width
                        display_height = int(display_width / aspect_ratio)
                    
                    display_frame = cv2.resize(frame, (display_width, display_height))
                    
                    # Convert to ImageTk format
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(display_frame))
                    
                    # Update canvas
                    self.camera_canvas.create_image(
                        canvas_width // 2,
                        canvas_height // 2,
                        image=self.photo,
                        anchor=tk.CENTER
                    )
                    
                    # Bring countdown label to front
                    self.countdown_label.lift()
            
            # Schedule next update
            self.after_id = self.window.after(10, self.update_camera)
    
    def start_countdown(self):
        """Start countdown for capturing gesture"""
        if self.game_running and not self.countdown_active:
            self.countdown_active = True
            self.countdown_value = 5
            self.update_countdown()
    
    def update_countdown(self):
        """Update countdown timer"""
        if self.countdown_active:
            if self.countdown_value > 0:
                self.countdown_label.configure(text=str(self.countdown_value))
                self.countdown_value -= 1
                self.after_id = self.window.after(1000, self.update_countdown)
            else:
                self.countdown_label.configure(text="Shoot!")
                self.after_id = self.window.after(1000, self.capture_and_process)
    
    def capture_and_process(self):
        """Capture and process the image"""
        if self.game_running:
            self.countdown_active = False
            self.countdown_label.configure(text="")
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                self.show_error("Failed to capture image")
                return
            
            # Process image in a separate thread to avoid blocking UI
            self.processing_active = True
            processing_thread = threading.Thread(target=self.process_image, args=(frame,))
            processing_thread.daemon = True
            processing_thread.start()
    
    def process_image(self, frame):
        """
        Process the captured image to detect gesture
        This is a placeholder - implement actual processing logic
        """
        # Simulate processing steps
        processing_steps = []
        
        # Step 1: Original image
        processing_steps.append(("Original", frame.copy()))
        
        # Step 2: Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processing_steps.append(("Grayscale", gray.copy()))
        
        # Step 3: Blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        processing_steps.append(("Blurred", blurred.copy()))
        
        # Step 4: Threshold
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processing_steps.append(("Thresholded", thresh.copy()))
        
        # Step 5: Find contours
        contour_img = frame.copy()
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        processing_steps.append(("Contours", contour_img))
        
        # Display processing steps
        self.display_processing_steps(processing_steps)
        
        # Get user gesture (simulated)
        import random
        from enum import Enum
        
        class Gesture(Enum):
            ROCK = 1
            PAPER = 2
            SCISSORS = 3
            LIZARD = 4
            SPOCK = 5
        
        class GameResult(Enum):
            WIN = 1
            LOSE = 2
            TIE = 3
        
        # Simulate gesture recognition
        gestures = [Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS]
        if self.game_mode.get() == "extended":
            gestures.extend([Gesture.LIZARD, Gesture.SPOCK])
        
        user_gesture = random.choice(gestures)
        computer_gesture = random.choice(gestures)
        
        # Determine winner
        result = self.determine_winner(user_gesture, computer_gesture)
        
        # Update UI with results
        self.window.after(0, lambda: self.update_results(user_gesture, computer_gesture, result))
    
    def determine_winner(self, user_gesture, computer_gesture):
        """
        Determine the winner based on the gestures
        This is a placeholder - implement actual game logic
        """
        from enum import Enum
        
        class GameResult(Enum):
            WIN = 1
            LOSE = 2
            TIE = 3
        
        # Tie
        if user_gesture == computer_gesture:
            return GameResult.TIE
        
        # Classic rock-paper-scissors rules
        if user_gesture.name == "ROCK":
            if computer_gesture.name == "SCISSORS":
                return GameResult.WIN
            elif computer_gesture.name == "PAPER":
                return GameResult.LOSE
            elif computer_gesture.name == "LIZARD":
                return GameResult.WIN
            elif computer_gesture.name == "SPOCK":
                return GameResult.LOSE
        
        elif user_gesture.name == "PAPER":
            if computer_gesture.name == "ROCK":
                return GameResult.WIN
            elif computer_gesture.name == "SCISSORS":
                return GameResult.LOSE
            elif computer_gesture.name == "LIZARD":
                return GameResult.LOSE
            elif computer_gesture.name == "SPOCK":
                return GameResult.WIN
        
        elif user_gesture.name == "SCISSORS":
            if computer_gesture.name == "PAPER":
                return GameResult.WIN
            elif computer_gesture.name == "ROCK":
                return GameResult.LOSE
            elif computer_gesture.name == "LIZARD":
                return GameResult.WIN
            elif computer_gesture.name == "SPOCK":
                return GameResult.LOSE
        
        # Extended rules for lizard and spock
        elif user_gesture.name == "LIZARD":
            if computer_gesture.name == "PAPER":
                return GameResult.WIN
            elif computer_gesture.name == "ROCK":
                return GameResult.LOSE
            elif computer_gesture.name == "SCISSORS":
                return GameResult.LOSE
            elif computer_gesture.name == "SPOCK":
                return GameResult.WIN
        
        elif user_gesture.name == "SPOCK":
            if computer_gesture.name == "SCISSORS":
                return GameResult.WIN
            elif computer_gesture.name == "ROCK":
                return GameResult.WIN
            elif computer_gesture.name == "PAPER":
                return GameResult.LOSE
            elif computer_gesture.name == "LIZARD":
                return GameResult.LOSE
        
        # Default
        return GameResult.TIE
    
    def update_results(self, user_gesture, computer_gesture, result):
        """Update the UI with the game results"""
        # Update gesture labels
        self.user_gesture_label.configure(text=user_gesture.name)
        self.computer_gesture_label.configure(text=computer_gesture.name)
        
        # Update result label and scores
        if result.name == "WIN":
            self.result_label.configure(text="YOU WIN!")
            self.user_score += 1
        elif result.name == "LOSE":
            self.result_label.configure(text="YOU LOSE!")
            self.computer_score += 1
        else:
            self.result_label.configure(text="TIE!")
        
        # Update score labels
        self.user_score_label.configure(text=str(self.user_score))
        self.computer_score_label.configure(text=str(self.computer_score))
        
        # Re-enable start button
        self.start_button.configure(state=tk.NORMAL)
    
    def display_processing_steps(self, processing_steps):
        """Display the image processing steps on the processing canvas"""
        # Get canvas dimensions
        canvas_width = self.processing_canvas.winfo_width()
        canvas_height = self.processing_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Clear canvas
        self.processing_canvas.delete("all")
        
        # Calculate grid layout
        num_steps = len(processing_steps)
        cols = 2
        rows = (num_steps + cols - 1) // cols
        
        # Calculate thumbnail size
        thumb_width = canvas_width // cols - 10
        thumb_height = canvas_height // rows - 40
        
        # Display each processing step
        for i, (title, image) in enumerate(processing_steps):
            # Calculate position
            row = i // cols
            col = i % cols
            
            x = col * (thumb_width + 10) + 5
            y = row * (thumb_height + 40) + 5
            
            # Resize image to thumbnail size
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image maintaining aspect ratio
            h, w = image.shape[:2]
            aspect = w / h
            
            if thumb_width / thumb_height > aspect:
                # Thumbnail is wider than image
                display_height = thumb_height
                display_width = int(display_height * aspect)
            else:
                # Thumbnail is taller than image
                display_width = thumb_width
                display_height = int(display_width / aspect)
            
            # Resize image
            display_img = cv2.resize(image, (display_width, display_height))
            
            # Convert to PhotoImage
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(display_img))
            
            # Keep reference to avoid garbage collection
            if not hasattr(self, 'processing_photos'):
                self.processing_photos = []
            self.processing_photos.append(photo)
            
            # Create image on canvas
            self.processing_canvas.create_image(
                x + thumb_width // 2,
                y + thumb_height // 2,
                image=photo,
                anchor=tk.CENTER
            )
            
            # Add title
            self.processing_canvas.create_text(
                x + thumb_width // 2,
                y + thumb_height + 10,
                text=title,
                font=("Arial", 10, "bold"),
                anchor=tk.CENTER
            )
    
    def show_error(self, message):
        """Show error message"""
        from tkinter import messagebox
        messagebox.showerror("Error", message)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_game()
        self.window.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = RockPaperScissorsGUI(root, "Rock Paper Scissors Game")


if __name__ == "__main__":
    main()