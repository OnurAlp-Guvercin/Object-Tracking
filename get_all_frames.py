import cv2
import os

# Specify the path to the video file and the output folder
video_path = "videos/test/test.mp4"
output_folder = "test_images"

# Create the output folder if it doesn't already exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a VideoCapture object to read in the video file
cap = cv2.VideoCapture(video_path)

# Loop over the frames in the video and extract each one
frame_count = 0
while cap.isOpened():
    # Read in the next frame
    ret, frame = cap.read()
    
    # If there are no more frames, break out of the loop
    if not ret:
        break
    
    # Construct the path to save the frame
    output_path = os.path.join(output_folder, '{:04d}.jpg'.format(frame_count))
    
    # Write the frame to the output path
    cv2.imwrite(output_path, frame)
    
    # Increment the frame count
    frame_count += 1

# Release the VideoCapture object and close any windows
cap.release()
cv2.destroyAllWindows()

print("Extracted and saved {} frames from the video".format(frame_count))
