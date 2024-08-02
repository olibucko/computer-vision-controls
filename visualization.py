# First step. Create function to visualise coordinate.
import pyglet
from pyglet import shapes
import random

# Construct viewing window
window = pyglet.window.Window()

# Construct graphical batch. For efficient drawing of many shapes.
batch = pyglet.graphics.Batch()

#Coordinates dictionary.
coords = {}

#Populate dictionary. 
for i in range(0,21):
    coords.update({i:shapes.Circle(x=i + i * 30, y=i + i * 30, radius=5, color=(50, 225, 30), batch=batch)})

# Start the window
@window.event
def on_draw():
    window.clear()
    batch.draw()

# # Event loop
def update_data(dt):
    new_coords = []
    for i in range(0,21):

        # Feed application live data
        coords[i].x = random.randint(0,600)
        coords[i].y = random.randint(0,600)

# pyglet.clock.schedule_interval(update_data, 0.05)
pyglet.app.run()
