import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Function to identify the letter based on hand landmarks
def identify_letter(hand_landmarks):
    """Identify the ASL letter based on hand landmarks."""
    
    # Get positions of landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Calculate distances
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
    hand_size = calculate_distance(wrist, middle_tip)
    dynamic_threshold = hand_size * 0.2  # Dynamic threshold based on hand size

    # Print for debugging purposes
    print("Thumb-Index Distance: ", thumb_index_dist)
    print("Thumb-Pinky Distance: ", thumb_pinky_dist)

    # Conditions for detecting each letter
    if index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y and thumb_tip.x > index_tip.x:
        return "A"  # Fist with thumb extended
    elif thumb_tip.x < index_tip.x < middle_tip.x < ring_tip.x < pinky_tip.x and all(
        [thumb_tip.y > finger.y for finger in [index_tip, middle_tip, ring_tip, pinky_tip]]
    ):
        return "B"  # Flat hand with fingers extended
    elif thumb_index_dist > dynamic_threshold and thumb_pinky_dist > dynamic_threshold:
        return "C"  # Hand forming a "C" shape
    elif thumb_index_dist > dynamic_threshold and all(
        [middle_tip.y > index_tip.y, ring_tip.y > middle_tip.y, pinky_tip.y > ring_tip.y]
    ):
        return "D"  # Thumb and index extended, other fingers curled
    elif thumb_index_dist < dynamic_threshold and thumb_pinky_dist < dynamic_threshold:
        return "E"  # Fingers curled, thumb wrapping over
    elif thumb_index_dist < dynamic_threshold and middle_tip.y < ring_tip.y < pinky_tip.y:
        return "F"  # Thumb and index close to form a circle
    elif thumb_index_dist > dynamic_threshold and thumb_tip.y < index_tip.y and middle_tip.y > index_tip.y:
        return "G"  # Thumb and index extended sideways
    elif thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y and ring_tip.y > middle_tip.y:
        return "H"  # Index and middle fingers extended
    elif thumb_pinky_dist > dynamic_threshold and all(
        [index_tip.y > middle_tip.y, middle_tip.y > ring_tip.y]
    ):
        return "I"  # Thumb and pinky extended, others curled
    elif thumb_pinky_dist > dynamic_threshold and pinky_tip.x < thumb_tip.x:
        return "J"  # Pinky makes a "J" motion (similar to I)
    elif thumb_index_dist > dynamic_threshold and index_tip.y < middle_tip.y and thumb_tip.y < index_tip.y:
        return "K"  # Thumb and index at 90 degrees, middle extended
    elif thumb_tip.x < index_tip.x and thumb_tip.y > index_tip.y:
        return "L"  # Thumb and index extended to form "L"
    elif thumb_tip.x < index_tip.x and thumb_tip.y > middle_tip.y > ring_tip.y:
        return "M"  # Thumb hidden under index and middle fingers
    elif thumb_tip.x < middle_tip.x and thumb_tip.y > ring_tip.y > pinky_tip.y:
        return "N"  # Thumb hidden under middle and ring fingers
    elif thumb_index_dist < dynamic_threshold and thumb_pinky_dist > dynamic_threshold:
        return "O"  # Thumb and index touching to form a circle
    elif thumb_tip.y < index_tip.y < middle_tip.y and ring_tip.y > middle_tip.y:
        return "P"  # Thumb and index extended downward
    elif thumb_tip.y < index_tip.y < middle_tip.y and wrist.y > index_tip.y:
        return "Q"  # Similar to G but with wrist rotation
    elif index_tip.x < middle_tip.x and ring_tip.y > middle_tip.y > pinky_tip.y:
        return "R"  # Index and middle fingers crossed
    elif thumb_tip.x > index_tip.x and thumb_tip.y > index_tip.y:
        return "S"  # Hand closed into a fist
    elif thumb_tip.x > index_tip.x and thumb_tip.y < index_tip.y:
        return "T"  # Thumb crosses over index finger
    elif index_tip.y < middle_tip.y and ring_tip.y > middle_tip.y:
        return "U"  # Index and middle fingers together, others curled
    elif calculate_distance(index_tip, middle_tip) > dynamic_threshold:
        return "V"  # Index and middle fingers spread apart
    elif calculate_distance(index_tip, middle_tip) > dynamic_threshold and calculate_distance(middle_tip, ring_tip) > dynamic_threshold:
        return "W"  # Index, middle, and ring fingers spread apart
    elif thumb_tip.x > index_tip.x and thumb_index_dist < dynamic_threshold:
        return "X"  # Index finger curled
    elif thumb_pinky_dist > dynamic_threshold and index_tip.y > middle_tip.y:
        return "Y"  # Thumb and pinky extended
    elif index_tip.x > wrist.x and thumb_tip.x < index_tip.x:
        return "Z"  # Draw "Z" with index finger

    return "Unknown"  # If no match, return "Unknown"

# Function to process the video frame
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_letter = identify_letter(hand_landmarks)
            cv2.putText(frame, f"Detected: {detected_letter}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

# Sign recognition application
def sign_recognition_app():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the webcam.")
            break
        frame = process_frame(frame)
        cv2.imshow("ASL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sign_recognition_app()
