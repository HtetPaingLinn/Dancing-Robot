import time
import cv2
import librosa
import numpy as np

# Function to detect beats in the audio file
def detect_beats(audio_file):
    y, sr = librosa.load(audio_file)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return beat_times, tempo

# Function to draw a simple dancing robot
def draw_robot(frame, movement_type=0):
    img_h, img_w = frame.shape[:2]
    
    # Robot body dimensions
    body_width, body_height = 100, 150
    body_top_left = (int(img_w / 2) - body_width // 2, int(img_h / 2) - body_height // 2)
    body_bottom_right = (int(img_w / 2) + body_width // 2, int(img_h / 2) + body_height // 2)
    
    # Draw the body (rectangle)
    cv2.rectangle(frame, body_top_left, body_bottom_right, (0, 255, 0), -1)
    
    # Draw the head (circle)
    head_center = (int(img_w / 2), body_top_left[1] - 50)
    cv2.circle(frame, head_center, 40, (0, 255, 0), -1)
    
    # Draw the arms
    left_arm_start = (body_top_left[0] - 50, body_top_left[1] + 30)
    right_arm_start = (body_bottom_right[0] + 50, body_top_left[1] + 30)
    
    if movement_type == 0:
        # Up-down movement for arms
        left_arm_end = (left_arm_start[0], left_arm_start[1] - 50)
        right_arm_end = (right_arm_start[0], right_arm_start[1] - 50)
    else:
        # Alternating movement for arms
        left_arm_end = (left_arm_start[0], left_arm_start[1] + 50)
        right_arm_end = (right_arm_start[0], right_arm_start[1] + 50)

    cv2.line(frame, left_arm_start, left_arm_end, (255, 0, 0), 5)
    cv2.line(frame, right_arm_start, right_arm_end, (255, 0, 0), 5)

    # Draw the legs
    left_leg_start = (body_top_left[0] + 20, body_bottom_right[1])
    right_leg_start = (body_bottom_right[0] - 20, body_bottom_right[1])
    
    if movement_type == 1:
        # Swing legs on beats
        left_leg_end = (left_leg_start[0] - 20, left_leg_start[1] + 70)
        right_leg_end = (right_leg_start[0] + 20, right_leg_start[1] + 70)
    else:
        left_leg_end = (left_leg_start[0], left_leg_start[1] + 70)
        right_leg_end = (right_leg_start[0], right_leg_start[1] + 70)

    cv2.line(frame, left_leg_start, left_leg_end, (255, 0, 0), 5)
    cv2.line(frame, right_leg_start, right_leg_end, (255, 0, 0), 5)

    return frame

# Function to synchronize robot's dance with detected beats
def sync_robot_to_music(audio_file):
    # Detect the beats in the music
    beat_times, tempo = detect_beats(audio_file)

    # Set up the video writer to save the dance sequence
    cap = cv2.VideoCapture(0)  # Use webcam feed as background (optional)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter('robot_dance.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
    
    # Initialize the timing
    start_time = time.time()
    current_beat = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we need to switch to the next dance movement based on the beat
        current_time = time.time() - start_time
        if current_beat < len(beat_times) and current_time >= beat_times[current_beat]:
            movement_type = current_beat % 2  # Alternate between two movement types
            current_beat += 1

        # Generate the robot dance and overlay on the frame
        robot_frame = draw_robot(frame, movement_type)
        
        # Display the robot dance
        cv2.imshow('Robot Dance', robot_frame)
        out.write(robot_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main function to run the robot dance generator
if __name__ == "__main__":
    audio_file = 'song.mp3'  # Replace with your audio file
    sync_robot_to_music(audio_file)
