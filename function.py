import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize Mediapipe Hands and Drawing Styles
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    """Detects hands in the image using Mediapipe Hands model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    """Draws landmarks on the image."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

def extract_keypoints(results):
    """Extracts 21 keypoints (x, y, z) from the hand landmarks."""
    if results.multi_hand_landmarks:
        # Extract keypoints for the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        return rh
    return np.zeros(21 * 3)  # Return a zeroed array if no hand is detected

# Path for exported data, numpy arrays
DATA_PATH = r'C:\Users\nooru\Desktop\Sign language\SignLanguageDetectionUsingML-main\AtoZ_3.1'

# List of actions (letters A-Z)
actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

# Number of sequences to record for each letter
no_sequences = 30

# Sequence length for each data sample
sequence_length = 30
