
import os
from pathlib import Path
import glob
from rasterstats import zonal_stats
import geopandas as gpd
from rasterio.merge import merge
import rasterio

proj_path = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS"

conf = os.path.join(proj_path, "predict_temp")
# polys = os.path.join(proj_path, "test_output.shp")
# gdf = gpd.read_file(polys)

allconfs = [os.path.join(conf, f) for f in os.listdir(conf) if f.endswith('conf.tif')]

raster_to_mosiac = []

for conf in allconfs:
    raster = rasterio.open(conf)
    raster_to_mosiac.append(raster)

mosaic, output = merge(raster_to_mosiac)

output_meta = raster.meta.copy()
output_meta.update(
    {"driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    }
)

output_path = os.path.join(proj_path, 'mosaic.tif')
with rasterio.open(output_path, "w", **output_meta) as m:
    m.write(mosaic)


