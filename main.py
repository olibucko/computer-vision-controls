from multiprocessing import Process, Array, Value
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyglet
from pyglet import shapes
import pyvolume

# Gesture codes
gesture_index = {0:"None", 1:"Closed_Fist", 2:"Open_Palm", 3:"Pointing_Up", 4:"Thumb_Down", 5:"Thumb_Up", 6:"Victory", 7:"ILoveYou"}

def gestures (model: str, num_hands: int, min_hand_detection_confidence: float,
         min_hand_presence_confidence: float, min_tracking_confidence: float,
         camera_id: int, width: int, height: int, array, gesture_code) -> None:

  # Capture video and set parameters
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

  # Variables for recognition logic
  recognition_result_list = []

  def save_result(result: vision.GestureRecognizerResult, unused_output_image: mp.Image, # type: ignore
                  timestamp_ms: int):

    recognition_result_list.append(result)

  # Start gesture recognizer
  base_options = python.BaseOptions(model_asset_path=model, delegate=python.BaseOptions.Delegate.GPU)
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

    # Read frame from camera
    success, image = cap.read()
    if not success:
      sys.exit("ERROR: Cannot read from webcam.")
    
    # Flip camera to correct orientation
    image = cv2.flip(image, 1)

    # Scale image for improved performance
    h, w = image.shape[:2]
    s_h, s_w = int(h), int(w)
    scaled_image = cv2.resize(image, (s_h, s_w), interpolation=cv2.INTER_LINEAR)

    # Convert scaled_image to RGB format as needed by MediaPipe framework
    rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Recognize gestures in the frame
    recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

    # If any hands are detected
    if recognition_result_list:

      # Enumerate detection results and extract information.
      for hand_index, hand_landmarks in enumerate(
        recognition_result_list[0].hand_landmarks):
        
        # Get gesture classification results
        if recognition_result_list[0].gestures:
          gesture = recognition_result_list[0].gestures[hand_index]
          category_name = gesture[0].category_name
          score = round(gesture[0].score, 2)

          # Assign gesture code to shared memory value
          for g_code in gesture_index:
             if gesture_index[g_code] == category_name:
                gesture_code.value = g_code

        # Assign hand coordinates to shared memory array
        i = 0
        
        for landmark in hand_landmarks:
          array[i] = landmark.x
          array[i+1] = landmark.y
          i = i + 2


      recognition_result_list.clear()

    if image is not None:
        cv2.imshow('gesture_recognition', image)          

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break
    
    time.sleep(1/30)

  recognizer.close()
  cap.release()
  cv2.destroyAllWindows()

def visualize (array, gesture_code):
  # Construct viewing window
  window = pyglet.window.Window(1280, 720)
  
  # Construct graphical batch. For efficient drawing of many shapes.
  batch = pyglet.graphics.Batch()

  # Dictionary of coordinates indicators drawn to screen.
  shapes_drawn = {}

  # Display what laebl is on the screen
  gesture_label = pyglet.text.Label('Program running.',
                          font_name='Times New Roman',
                          font_size=24,
                          x=window.width-160, y=window.height-40,
                          anchor_x='center', anchor_y='center')

  # Populate shapes_drawn 
  for i in range(0,21):
      shapes_drawn.update({i:shapes.Circle(x=i + i * 30, y=i + i * 30, radius=5, color=(50, 225, 30), batch=batch)})

  # Coordinates are between 0-1. Scale up to representable pixel size.
  scale_factor = 650

  # Rendering Offset
  x_offset = 400
  y_offset= 600

  # Start the window
  @window.event
  def on_draw():
      window.clear()
      batch.draw()
      gesture_label.draw()

  # Event loop
  def update_data(dt):

    # Get latest coordinates from share array (if any)
    coordinates = array[:]

    if coordinates:
      i = 0
      j = 1

      # Assign new coordinates to drawn shapes
      for index in shapes_drawn:
        x = coordinates[i] * scale_factor
        y = coordinates[j] * scale_factor

        shapes_drawn[index].x = x + x_offset
        shapes_drawn[index].y = -y + y_offset

        # Prevent out of range errors
        if index <= 20:
          i = i + 2
          j = j + 2

      # Assign gesture name to gesture label
      g_code = gesture_code.value
      gesture_label.text = gesture_index[g_code]

  pyglet.clock.schedule_interval(update_data, 1/30)
  pyglet.app.run()

def controls (array, gesture_code):
   while True:
      # coordinates = array[:]
      g_code = gesture_code.value
      
      # If gesture is recognized
      if g_code in gesture_index:
        if g_code == 1:
          pyvolume.custom(percent=0) # Mute control
        elif g_code == 5:
           pyvolume.custom(percent=75) # Volume on control

      time.sleep(1/30)


if __name__ == '__main__':

  # Define shared memory space between processes
  array = Array('d', range(42))
  gesture_code = Value('i')

  # Start subprocesses
  recognizer = Process(target=gestures, args=('resources/gesture_recognizer.task', 2, 0.8, 0.8, 0.8, 0, 640, 480, array, gesture_code))
  recognizer.start()

  visualizer = Process(target=visualize, args=(array, gesture_code))
  visualizer.start()

  controller = Process(target=controls, args=(array, gesture_code))
  controller.start()

  recognizer.join()
  visualizer.join()
  controller.join()