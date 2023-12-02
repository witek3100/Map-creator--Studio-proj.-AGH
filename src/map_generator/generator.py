import os
import sys
import cv2
from math import floor
import numpy as np
from src.ml.scripts.predict import predict
from PIL import Image
from config import BASE_DIR
from scipy.ndimage import measurements, zoom
from numba import jit


@jit(nopython=True)
def is_inside(x, y, poly):
    n = len(poly)
    inside = False
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def generate_map(image, model='model1'):

    tile_size = 50

    image_size = tuple(floor(x / 50) for x in image.size)

    tiles = np.empty(image_size, dtype=object)

    for i in range(0, tiles.shape[0]):
        for j in range(0, tiles.shape[1]):
            x = i * tile_size
            y = j * tile_size
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles[i, j] = np.array(tile)

    predictions = predict('model1', tiles.flatten())

    pred_map = np.array(predictions).reshape(image_size)

    pred_map, clusters = measurements.label(pred_map)

    clusters_to_filter = [cluster for cluster in np.unique(pred_map) if np.count_nonzero(pred_map == cluster) <= 2]
    mask = np.isin(pred_map, clusters_to_filter)
    pred_map[mask] = 0
    pred_mapT = pred_map.T

    areas = {i: [(x, y) for x, y in zip(np.where(pred_map == i)[0], np.where(pred_map == i)[1])] for i in
             np.unique(pred_map)[1:]}

    for key in areas.keys():
        row = areas[key]
        new_row = []
        for x in row:
            new_row.extend([(x[0], x[1]), (x[0] + 1, x[1]), (x[0], x[1] + 1), (x[0] + 1, x[1] + 1)])
        areas[key] = list(set(new_row))

    window_size = 100
    areass = [tile_size * np.array(areas[area]) for area in areas.keys()]
    new_areas = []
    for i in range(0, image.size[0], window_size):
        for j in range(0, image.size[1], window_size):

            for area in areass:
                new_area = []
                new_area = [x for x in area if i <= x[0] <= i + window_size and j <= x[1] <= j + window_size]
                if len(new_area):
                    new_areas.append(new_area)

    new_areas = [cv2.convexHull(np.array(area)) for area in new_areas]
    new_areas = [np.array([x[0] for x in area]) for area in new_areas]

    #########################
    map = np.zeros(image.size)
    for i in range(map.shape[0]):
        print(f'{i}/{map.shape[0]}')
        for j in range(map.shape[1]):
            for area in new_areas:
                if is_inside(i, j, area):
                    map[i, j] = 200
    result = np.stack((map.T, np.array(image.split()[1]), np.array(image.split()[2])), axis=0)
    result = result.T.astype('uint8')
    result = np.transpose(result, (1, 0, 2))
    result = Image.fromarray(result)
    result.show()
    #########################

    return new_areas


if __name__ == '__main__':
    try:
        image_name = sys.argv[1]
        try:
            image = Image.open(os.path.join(BASE_DIR, f'data/satellite_images/{image_name}'))
            generate_map(image.crop((0, 0, 6000, 6000)))

        except FileNotFoundError:
            raise Exception("Can't find this image")
    except IndexError:
        raise Exception('No image provided')