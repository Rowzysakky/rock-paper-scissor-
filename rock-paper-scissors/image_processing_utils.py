"""
Image Processing Utilities for Rock Paper Scissors Game
CS402.3 Coursework

This module contains utility functions for image processing and hand gesture recognition.
It implements more advanced techniques for segmentation, background removal, and gesture classification.
"""

import cv2
import numpy as np
import mediapipe as mp


class HandGestureRecognizer:
    """
    Class for recognizing hand gestures using MediaPipe
    MediaPipe provides more accurate hand landmark detection than basic contour analysis
    """
    
    def __init__(self):
        """Initialize the hand gesture recognizer"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def detect_landmarks(self, image):
        """
        Detect hand landmarks in the image
        Returns:
            landmarks (list): List of detected hand landmarks
            processed_image (numpy.ndarray): Image with landmarks drawn
        """
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = self.hands.process(image_rgb)
        
        # Draw hand landmarks on the image
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return results.multi_hand_landmarks, annotated_image
    
    def recognize_gesture(self, landmarks):
        """
        Recognize the gesture based on hand landmarks
        Args:
            landmarks: MediaPipe hand landmarks
        Returns:
            gesture: The recognized gesture
        """
        from enum import Enum
        
        class Gesture(Enum):
            ROCK = 1
            PAPER = 2
            SCISSORS = 3
            UNKNOWN = 4
        
        if not landmarks:
            return Gesture.UNKNOWN
        
        # Get the first hand's landmarks
        hand_landmarks = landmarks[0]
        
        # Extract finger landmarks
        # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Extract palm landmarks
        wrist = hand_landmarks.landmark[0]
        index_mcp = hand_landmarks.landmark[5]  # Index finger MCP joint
        middle_mcp = hand_landmarks.landmark[9]  # Middle finger MCP joint
        ring_mcp = hand_landmarks.landmark[13]  # Ring finger MCP joint
        pinky_mcp = hand_landmarks.landmark[17]  # Pinky finger MCP joint
        
        # Calculate the distance of each fingertip from the palm
        # A finger is considered extended if the fingertip is far from the palm
        thumb_extended = self._is_finger_extended(thumb_tip, wrist, index_mcp)
        index_extended = self._is_finger_extended(index_tip, wrist, index_mcp)
        middle_extended = self._is_finger_extended(middle_tip, wrist, middle_mcp)
        ring_extended = self._is_finger_extended(ring_tip, wrist, ring_mcp)
        pinky_extended = self._is_finger_extended(pinky_tip, wrist, pinky_mcp)
        
        # Count extended fingers
        extended_fingers = sum([
            thumb_extended, 
            index_extended, 
            middle_extended, 
            ring_extended, 
            pinky_extended
        ])
        
        # Recognize gesture based on finger positions
        # Rock: fist (no extended fingers or just thumb)
        # Paper: all fingers extended
        # Scissors: index and middle fingers extended, others curled
        
        if extended_fingers <= 1:
            return Gesture.ROCK
        
        if extended_fingers >= 4:
            return Gesture.PAPER
        
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return Gesture.SCISSORS
        
        # Default if no specific pattern is recognized
        return Gesture.UNKNOWN
    
    def _is_finger_extended(self, fingertip, wrist, mcp_joint):
        """
        Determine if a finger is extended based on its position relative to the palm
        """
        # Calculate the distance from fingertip to wrist
        wrist_to_tip = self._euclidean_distance(fingertip, wrist)
        
        # Calculate the distance from MCP joint to wrist
        wrist_to_mcp = self._euclidean_distance(mcp_joint, wrist)
        
        # A finger is considered extended if the fingertip is significantly 
        # further from the wrist than the MCP joint
        return wrist_to_tip > (wrist_to_mcp * 1.5)
    
    def _euclidean_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points"""
        return np.sqrt(
            (point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2 + 
            (point1.z - point2.z) ** 2
        )


class BackgroundRemover:
    """
    Class for removing the background from images
    This improves hand gesture recognition by isolating the hand
    """
    
    def __init__(self):
        """Initialize the background remover"""
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
    
    def remove_background(self, image):
        """
        Remove the background from an image
        Returns:
            foreground_mask (numpy.ndarray): Binary mask of the foreground
            result (numpy.ndarray): Image with background removed
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)
        
        # Apply morphological operations to improve the mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=fg_mask)
        
        return fg_mask, result
    
    def remove_background_static(self, image):
        """
        Remove background using static methods (useful for single images)
        Returns:
            foreground_mask (numpy.ndarray): Binary mask of the foreground
            result (numpy.ndarray): Image with background removed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=fg_mask)
        
        return fg_mask, result


class ImageEnhancer:
    """
    Class for enhancing image quality to improve gesture recognition
    """
    
    @staticmethod
    def adjust_brightness_contrast(image, alpha=1.5, beta=0):
        """
        Adjust brightness and contrast of an image
        Args:
            image (numpy.ndarray): Input image
            alpha (float): Contrast control (1.0 means no change)
            beta (int): Brightness control (0 means no change)
        Returns:
            adjusted (numpy.ndarray): Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    
    @staticmethod
    def sharpen(image):
        """
        Sharpen an image to enhance edges
        Args:
            image (numpy.ndarray): Input image
        Returns:
            sharpened (numpy.ndarray): Sharpened image
        """
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    @staticmethod
    def reduce_noise(image):
        """
        Reduce noise in an image
        Args:
            image (numpy.ndarray): Input image
        Returns:
            denoised (numpy.ndarray): Denoised image
        """
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised


class ColorSegmenter:
    """
    Class for segmenting images based on skin color
    This can help isolate the hand in the image
    """
    
    @staticmethod
    def extract_skin(image):
        """
        Extract skin regions from an image based on color
        Args:
            image (numpy.ndarray): Input image in BGR format
        Returns:
            mask (numpy.ndarray): Binary mask of skin regions
            result (numpy.ndarray): Image with only skin regions
        """
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create a binary mask for skin regions
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # Apply morphological operations to improve the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return mask, result


# Utility functions for visualizing and displaying images
def create_processing_montage(images, titles=None, cols=2):
    """
    Create a montage of images for visualization
    Args:
        images (list): List of images
        titles (list): List of titles for the images
        cols (int): Number of columns in the montage
    Returns:
        montage (numpy.ndarray): Montage of images
    """
    # Calculate the number of rows needed
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Create a blank canvas for the montage
    # Find the maximum dimensions of all images
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # Create the montage with margin
    margin = 10
    montage_height = rows * (max_height + margin) - margin
    montage_width = cols * (max_width + margin) - margin
    montage = np.ones((montage_height, montage_width, 3), dtype=np.uint8) * 255
    
    # Place each image in the montage
    for i, image in enumerate(images):
        # Calculate position
        row = i // cols
        col = i % cols
        
        y = row * (max_height + margin)
        x = col * (max_width + margin)
        
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Make sure image is BGR
        h, w = image.shape[:2]
        montage[y:y+h, x:x+w] = image
        
        # Add title if provided
        if titles and i < len(titles):
            cv2.putText(montage, titles[i], (x, y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return montage