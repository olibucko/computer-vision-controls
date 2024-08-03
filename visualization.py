# First step. Create function to visualise coordinate.
import pyglet
from pyglet import shapes
import json
import shutil

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

# # Event loop
def update_data(dt):

    # Obtain raw data from file
    shutil.copyfile("landmarks.json", "visualise_data.json")

    f = open("visualise_data.json", "r")
    raw_data = json.load(f)
    f.close()
    
    # Name of gesture.
    gesture = raw_data[0]

    # Dictionary of coordinates.
    # Format as {"Reference ID": [X-Value, Y-Value], "Reference ID": [X-Value, Y-Value]}
    coordinates = raw_data[1]

    # Coordinates are between 0-1. Scale up to representable pixel size.
    scale_factor = 650

    for key in coordinates:
        
        value = coordinates[key]

        # Update each shape with new coordinates
        shapes_drawn[int(key)].x = value[0] * scale_factor
        shapes_drawn[int(key)].y = value[1] * scale_factor

pyglet.clock.schedule_interval(update_data, 0.25)
pyglet.app.run()