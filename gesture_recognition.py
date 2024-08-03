import argparse
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# Global vars to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
LANDMARKS = []

def run (model: str, num_hands: int, min_hand_detection_confidence: float,
         min_hand_presence_confidence: float, min_tracking_confidence: float,
         camera_id: int, width: int, height: int) -> None:
  
  global LANDMARKS
  
  # Capture video and set parameters
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

  # Parameters for visualization
  fps_avg_frame_count = 10

  # Variables for recognition logic
  recognition_result_list = []

  def save_result(result: vision.GestureRecognizerResult, unused_output_image: mp.Image,
                  timestamp_ms: int):
    global FPS, COUNTER, START_TIME
  
    if COUNTER % fps_avg_frame_count == 0:
      FPS = fps_avg_frame_count / (time.time() - START_TIME)
      START_TIME = time.time()

    recognition_result_list.append(result)
    COUNTER += 1

  # Start gesture recognizer
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.GestureRecognizerOptions(base_options=base_options,
                                          running_mode=vision.RunningMode.LIVE_STREAM,
                                          num_hands=num_hands,
                                          min_hand_detection_confidence=min_hand_detection_confidence,
                                          min_hand_presence_confidence=min_hand_presence_confidence,
                                          min_tracking_confidence=min_tracking_confidence,
                                          result_callback=save_result)
  
  # Recognizer instance to be referenced.
  recognizer = vision.GestureRecognizer.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():

    # Process every second frame for performance optimization.
    PROCESS_FRAME = True

    success, image = cap.read()
    if not success:
      sys.exit("ERROR: Cannot read from webcam.")
    
    image = cv2.flip(image, 1)

    # Get dimensions of image
    h, w = image.shape[:2]

    # Scale them down to 1/2
    s_h, s_w = int(h/2), int(w/2)

    # Scale the image using 1/2 dimensions
    scaled_image = cv2.resize(image, (s_h, s_w), interpolation=cv2.INTER_LINEAR)

    # Convert scaled_image to RGB format as needed by MediaPipe framework
    rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    if PROCESS_FRAME:
      # Recognize gestures in the frame
      recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

      # Set current_frame
      current_frame = image

      if recognition_result_list:

        # Access landmarks via enumeration. As per MediaPipe spec.
        for hand_index, hand_landmarks in enumerate(
          recognition_result_list[0].hand_landmarks):
          
          # Get gesture classification results
          if recognition_result_list[0].gestures:
            gesture = recognition_result_list[0].gestures[hand_index]
            category_name = gesture[0].category_name
            score = round(gesture[0].score, 2)

        
        # To access data use -> recognition_result_list[0].hand_landmarks[0][0].x, drop one of the [0] selectors when looping.
        # DATA FORMAT: NormalizedLandmark(x=value, y=value, z=value, visibility=value, presence=value)

        coords = {}
        i = 0

        try: 
          for data in recognition_result_list[0].hand_landmarks[0]:
            vector = []
            vector.append(data.x)
            vector.append(data.y)

            id = i
            i += 1

            # Format -> ID : [0.57253536 (x coordinate), 0345463242 (y coordinate)]
            coords.update({id:vector})
          
          print(coords)
        except:
          print("Could not perform coordinate extraction.")
          
        final_output_data = []

        try:
          final_output_data.append(category_name)
        except UnboundLocalError:
          final_output_data.append("None")
        
        try:
          final_output_data.append(coords)
        except UnboundLocalError:
          final_output_data.append("None")
      
        f = open("landmarks.txt", "w")
        f.write(str(final_output_data))
        f.close()

        recognition_result_list.clear()

    if image is not None:
        cv2.imshow('gesture_recognition', image)          
    
    # Process every second frame for performance optimization.
    PROCESS_FRAME = not PROCESS_FRAME

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break

  recognizer.close()
  cap.release()
  cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of gesture recognition model.',
      required=False,
      default='resources/gesture_recognizer.task')
  parser.add_argument(
      '--numHands',
      help='Max number of hands that can be detected by the recognizer.',
      required=False,
      default=2)
  parser.add_argument(
      '--minHandDetectionConfidence',
      help='The minimum confidence score for hand detection to be considered '
           'successful.',
      required=False,
      default=0.5)
  parser.add_argument(
      '--minHandPresenceConfidence',
      help='The minimum confidence score of hand presence score in the hand '
           'landmark detection.',
      required=False,
      default=0.5)
  parser.add_argument(
      '--minTrackingConfidence',
      help='The minimum confidence score for the hand tracking to be '
           'considered successful.',
      required=False,
      default=0.5)
  # Finding the camera ID can be very reliant on platform-dependent methods.
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.numHands), args.minHandDetectionConfidence,
      args.minHandPresenceConfidence, args.minTrackingConfidence,
      int(args.cameraId), args.frameWidth, args.frameHeight)

if __name__ == '__main__':
  main()
