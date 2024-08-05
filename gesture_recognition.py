from multiprocessing import Process, Array
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyglet
from pyglet import shapes

# Global variables
COUNTER, FPS = 0, 0
START_TIME = time.time()

def gestures (model: str, num_hands: int, min_hand_detection_confidence: float,
         min_hand_presence_confidence: float, min_tracking_confidence: float,
         camera_id: int, width: int, height: int, array) -> None:

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

    success, image = cap.read()
    if not success:
      sys.exit("ERROR: Cannot read from webcam.")
    
    image = cv2.flip(image, 0)

    # Get dimensions of image
    h, w = image.shape[:2]

    # Scale them down to 1/2
    s_h, s_w = int(h/2), int(w/2)

    # Scale the image using 1/2 dimensions
    scaled_image = cv2.resize(image, (s_h, s_w), interpolation=cv2.INTER_LINEAR)

    # Convert scaled_image to RGB format as needed by MediaPipe framework
    rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Recognize gestures in the frame
    recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

    # If callback from recognizer saved any results (detected hands)
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

      i = 0

      # 21 landmarks, 42 total entries
      try: 
        for data in recognition_result_list[0].hand_landmarks[0]:

          array[i] = data.x
          array[i+1] = data.y
          
          i = i + 2

      except:
        print("Could not perform coordinate extraction.")

      recognition_result_list.clear()

    if image is not None:
        cv2.imshow('gesture_recognition', image)          

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break
    
    time.sleep(1/60)

  recognizer.close()
  cap.release()
  cv2.destroyAllWindows()

def visualize (array):

  # Construct viewing window
  window = pyglet.window.Window()

  # Construct graphical batch. For efficient drawing of many shapes.
  batch = pyglet.graphics.Batch()

  #Coordinates dictionary.
  shapes_drawn = {}

  #Populate dictionary. 
  for i in range(0,21):
      shapes_drawn.update({i:shapes.Circle(x=i + i * 30, y=i + i * 30, radius=5, color=(50, 225, 30), batch=batch)})

  # Start the window
  @window.event
  def on_draw():
      window.clear()
      batch.draw()

  # Event loop
  def update_data(dt):

          # List of coordinates
          coordinates = array[:]

          # Coordinates are between 0-1. Scale up to representable pixel size.
          scale_factor = 650

          i = 0
          j = 1

          # Looping 21 times instead
          for index in shapes_drawn:
            x = coordinates[i] * scale_factor
            y = coordinates[j] * scale_factor

            shapes_drawn[index].x = x
            shapes_drawn[index].y = y

            if index <= 20:
              i = i + 2
              j = j + 2

  pyglet.clock.schedule_interval(update_data, 1/60)
  pyglet.app.run()

if __name__ == '__main__':

  # Define shared memory spaces between processes
  array = Array('d', range(42))

  recognizer = Process(target=gestures, args=('resources/gesture_recognizer.task', 2, 0.5, 0.5, 0.5, 0, 640, 480, array))
  recognizer.start()

  visualizer = Process(target=visualize, args=(array,))
  visualizer.start()

  recognizer.join()
  visualizer.join()