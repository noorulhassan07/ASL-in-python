import os
import cv2

# Directory path (update to match your actual path)
directory = r'C:\Users\nooru\Desktop\Sign language\SignLanguageDetectionUsingML-main\AtoZ_3.1'

# Ensure all letter directories exist
if not os.path.exists(directory):
    os.makedirs(directory)
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    letter_dir = os.path.join(directory, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize image count for each letter
count = {letter.lower(): len(os.listdir(os.path.join(directory, letter))) for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

print("Press the respective key (A-Z) to capture an image for that letter.")
print("Press 'ESC' to exit the program.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error accessing webcam. Exiting.")
        break

    # Define ROI (Region of Interest)
    start_x, start_y = 0, 40
    end_x, end_y = 300, 400
    roi = frame[start_y:end_y, start_x:end_x]

    # Draw rectangle for ROI
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

    # Display frame and ROI
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)

    # Key press handling
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # Exit on 'ESC' key
        print("Exiting the program.")
        break

    # Capture images for each key (A-Z)
    for letter in "abcdefghijklmnopqrstuvwxyz":
        if interrupt & 0xFF == ord(letter):
            save_path = os.path.join(directory, letter.upper(), f"{count[letter]}.jpg")
            cv2.imwrite(save_path, roi)
            print(f"Captured: {letter.upper()} -> {count[letter]}.jpg")
            count[letter] += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
