import math
import pickle


import numpy as np
import pyproj
import rasterio
from tqdm import tqdm

city_bounds = np.load(
    "city_bounds.npy"
)  # (MIN_LON, MIN_LAT, MAX_LON, MAX_LAT) for each city
world_pop = rasterio.open(
    "/data/population/raw/ppp_2019_1km_Aggregated.tif"  # path to WorldPop data
)
transform = world_pop.transform

world_pop_grid_count = []
bar = tqdm(total=len(city_bounds))
for bound in city_bounds:
    MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = bound
    lon = (MIN_LON + MAX_LON) / 2
    zone = (math.floor((lon + 180) / 6) % 60) + 1
    proj = pyproj.Proj(
        f"+proj=utm +zone={zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )
    MINX, MAXY = proj(MIN_LON, MAX_LAT)
    MAXX, MINY = proj(MAX_LON, MIN_LAT)

    GRID_SIZE = 1000
    GRID_NUM_X = int((MAXX - MINX) / GRID_SIZE) + 1
    GRID_NUM_Y = int((MAXY - MINY) / GRID_SIZE) + 1

    pop_gird_count = np.zeros((GRID_NUM_Y, GRID_NUM_X))
    for gx in range(GRID_NUM_X):
        for gy in range(GRID_NUM_Y):
            min_x, max_x = MINX + gx * GRID_SIZE, MINX + (gx + 1) * GRID_SIZE
            max_y, min_y = MAXY - gy * GRID_SIZE, MAXY - (gy + 1) * GRID_SIZE
            min_lon, min_lat = proj(min_x, min_y, inverse=True)
            max_lon, max_lat = proj(max_x, max_y, inverse=True)
            row_min, col_min = ~transform * (min_lon, max_lat)
            row_max, col_max = ~transform * (max_lon, min_lat)
            row_min, row_max, col_min, col_max = (
                math.floor(row_min),
                math.ceil(row_max),
                math.floor(col_min),
                math.ceil(col_max),
            )
            pop_data = world_pop.read(
                1, window=((col_min, col_max), (row_min, row_max))
            )
            pop_data[pop_data < 0] = 0
            pop_gird_count[gy, gx] = pop_data.sum()
    world_pop_grid_count.append(pop_gird_count)
    bar.update(1)
bar.close()

with open("world_pop_grid_count.pkl", "wb") as f:
    pickle.dump(world_pop_grid_count, f)
