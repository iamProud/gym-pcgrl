import os
from PIL import Image

folder = 'shared_runs/5x5/solve-example/5'

# load all images
images = []
for filename in os.listdir(folder):
    if filename.endswith(".png"):
        images.append(filename)

# sort them by name before .png ending
images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
print(images)

# create frames
frames = []
for i in range(len(images)):
    filename = os.path.join(folder, images[i])
    frames.append(Image.open(filename))

    # extend last frame
    if i == len(images) - 1:
        for _ in range(15):
            frames.append(Image.open(filename))

frames[0].save('solve.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
