from shapely.geometry import Polygon
import geopandas as gpd
from config import BASE_DIR
import os


class Map:
    def __init__(self, areas):
        self.name = 'map1.geojson'
        self.areas = areas

    def to_geojson(self):
        geometry = [Polygon(cords) for cords in self.areas]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
        gdf.to_file(os.path.join(BASE_DIR, f'data/maps/{self.name}'), driver='GeoJSON')

    def to_gpx(self):
        pass

    def show(self):
        pass
