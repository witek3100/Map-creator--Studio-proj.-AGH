import os
import math
from PIL import Image
from config import BASE_DIR
import numpy as np
import matplotlib.pyplot as plt
from src.ml.scripts.predict import predict

image = Image.open(os.path.join(BASE_DIR, 'data/satellite_images/ld.tif'))

tile_size = 50

image_size = list(map(lambda x : math.floor(x / 50), image.size))

tiles = np.empty(image_size, dtype=object)
print(image_size)

for i in range(0, tiles.shape[0]):
    for j in range(0, tiles.shape[1]):
        x = i * tile_size
        y = j * tile_size
        tile = image.crop((x, y, x + tile_size, y + tile_size))
        tiles[i, j] = np.array(tile)

predictions = predict('model1', tiles.flatten())

pred_map = np.array(predictions).reshape(image_size)
print(pred_map.shape)

for i in range(0, pred_map.shape[0]):
    for j in range(0, pred_map.shape[1]):
        x = i * tile_size
        y = j * tile_size

        if pred_map[i, j]:
            region = image.crop((x, y, x + tile_size, y + tile_size))
            red_channel = region.split()[0]
            red_channel = red_channel.point(lambda p: p * 1.5)
            new_region = Image.merge("RGB", (red_channel, region.split()[1], region.split()[2]))
            image.paste(new_region, (x, y, x + tile_size, y + tile_size))

image.show()


