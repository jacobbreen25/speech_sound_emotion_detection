import sounddevice as sd
import numpy as np
import threading
import time
from network_model import FullyConnectedNN
from generate_training_data import generate_features
import torch
from os.path import join
import cv2

# Configuration
SAMPLE_RATE = 22050  # Hz
BUFFER_SECONDS = 10  # Seconds of audio to store in the buffer
FRAME_SIZE = 1024  # Number of samples per frame

# Circular buffer to store audio data
buffer = np.zeros(BUFFER_SECONDS * SAMPLE_RATE, dtype=np.float32)
write_index = 0
buffer_lock = threading.Lock()

# Initialize position and step size for the moving element
slider_x = 50
slider_step = 25

def display_fullscreen(prediction_label, outputs):
    """
    Display a full-screen visualization with large text and colored rectangles.
    """
    global slider_x, slider_step
    # Set up the display parameters
    screen_width, screen_height = 1920, 1080  # Adjust as needed
    bg_color = (50, 50, 50)  # Background color (dark gray)
    text_color = (255, 255, 255)  # Text color (white)
    highlight_color = ( 102,255, 102)  # Color for the highest output (green)
    bar_color = ( 51, 153, 255)  # Default color for bars (red)

    # Create a blank image for the full-screen display
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img[:] = bg_color

    # Draw the prediction label in the center
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 5
    text = f"{prediction_label}"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (screen_width - text_size[0]) // 2
    text_y = (screen_height // 3)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Min-max normalization
    min_val = outputs.min()
    max_val = outputs.max()
    outputs = (outputs - min_val) / (max_val - min_val)
    # Draw rectangles for outputs
    num_outputs = len(outputs)
    bar_width = screen_width // (2 * num_outputs)
    bar_spacing = screen_width // num_outputs
    bar_height_max = screen_height // 3
    max_output_index = np.argmax(outputs)
    
    for i, output in enumerate(outputs):
        bar_height = int(output * bar_height_max)
        x1 = (i * bar_spacing) + (bar_spacing - bar_width) // 2
        y1 = screen_height - 100 - bar_height
        x2 = x1 + bar_width
        y2 = screen_height - 100

        color = highlight_color if i == max_output_index else bar_color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)  # Filled rectangle

    # Draw a moving rectangle at the bottom of the screen
    rectangle_width = 250
    rectangle_height = 20
    y_pos = screen_height - rectangle_height - 10  # Position near the bottom
    color = (0, 240, 0)  # Green rectangle
    
    # Draw the rectangle
    border = 50
    cv2.rectangle(img, (slider_x, y_pos), (slider_x + rectangle_width, y_pos + rectangle_height), color, -1)
    # Update the position of the rectangle
    slider_x += slider_step
    if slider_x + rectangle_width + border > screen_width:  # Wrap around when it reaches the edge
        slider_x = border
    
    # Display the image
    cv2.namedWindow("Real-Time Prediction", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Real-Time Prediction", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Real-Time Prediction", img)
    cv2.waitKey(1)  # Refresh frame


def audio_callback(indata, frames, time_info, status):
    """Callback function to process audio input."""
    global buffer, write_index
    if status:
        print(f"Audio callback error: {status}")
    
    with buffer_lock:
        end_index = write_index + frames
        if end_index < len(buffer):
            buffer[write_index:end_index] = indata[:, 0]  # Use the first channel
        else:
            # Wrap around the buffer
            split = len(buffer) - write_index
            buffer[write_index:] = indata[:split, 0]
            buffer[:end_index % len(buffer)] = indata[split:, 0]
        write_index = end_index % len(buffer)

# Start the audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
stream.start()

def get_last_second():
    """Retrieve the last second of audio data from the buffer."""
    global buffer, write_index
    with buffer_lock:
        start_index = (write_index - SAMPLE_RATE) % len(buffer)
        if start_index < 0:
            start_index += len(buffer)
        if start_index + SAMPLE_RATE <= len(buffer):
            return buffer[start_index:start_index + SAMPLE_RATE]
        else:
            return np.concatenate((buffer[start_index:], buffer[:start_index + SAMPLE_RATE - len(buffer)]))

print("Loading Model...")
num_features = 6336
model = FullyConnectedNN(num_features)
model_dir = "./models"
model_name = "model_v2-longtrain.pth"
# Load only the state dictionary safely
state_dict = torch.load(join(model_dir,model_name), weights_only=True)
# Load it into your model
model.load_state_dict(state_dict)
# Set the model to evaluation mode
model.eval()  
print("Model loaded.")
labels = ["silence", "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]



# Background loop to retrieve and process audio
def background_processing():
    global labels
    while True:
        audio_data = get_last_second()
        # print(f"Retrieved {len(audio_data)} samples.")
        features = generate_features(audio_data,SAMPLE_RATE)
        # print(f"Features: {len(features)}")
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        outputs = model(feature_tensor)
        output_array = outputs.detach().numpy()
        prediction = torch.argmax(outputs, dim=0)
        print(f"Prediction: {prediction} - {labels[prediction]}")
        display_fullscreen(labels[prediction], output_array)
        time.sleep(0.5)  # Process every second

# Start the background processing thread
thread = threading.Thread(target=background_processing, daemon=True)
thread.start()

print("Listening to microphone. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
    stream.stop()
    stream.close()
