import os
import sys
from PIL import Image
from config import BASE_DIR

if __name__ == "__main__":
    image_path = os.path.join(BASE_DIR, f'data/raw/{sys.argv[1]}')
    image = Image.open(image_path)

    tile_size = 50

    image_num = len(os.listdir(os.path.join(BASE_DIR, 'data/raw/maps'))) + 1

    for i in range(0, image.size[0], tile_size):
        for j in range(0, image.size[1], tile_size):
            if i + tile_size < image.size[0] and j + tile_size < image.size[1]:
                slice = image.crop((i, j, i + tile_size, j + tile_size))
                slice.save(f'maps/{image_num}.tif')
                image_num += 1
