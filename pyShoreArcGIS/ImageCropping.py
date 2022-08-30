
import arcpy
import rasterio
import geopandas as gpd
from itertools import product
import shutil
import os
from pathlib import Path
import shapely
import shutil


# Image Cropping
def get_tiles(ds, width=256, height=256):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = rasterio.windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = rasterio.windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = rasterio.windows.transform(window, ds.transform)
        yield window, transform


def get_tile_geom(tile_tif):

    rds = rasterio.open(tile_tif)
    minx, miny, maxx, maxy = rds.bounds
    geometry = shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)

    return geometry



# Input parameters
# data_dir = arcpy.GetParameterAsText(0)
# geo_data = arcpy.GetParameterAsText(1)
# outputfolder = arcpy.GetParameterAsText(2)

data_dir = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\data"
geo_data = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\data\cusp_2018_selected.shp"
outputfolder = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\crop"
Path(outputfolder).mkdir(exist_ok=True, parents=True)

N = 256

gdf = gpd.read_file(geo_data)
# arcpy.AddMessage("Processing: {}".format(gdf))

all_tiles = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('tif')]

for tile in all_tiles:

    tile_name = os.path.basename(tile).split('.')[0]
    output_filename = tile_name + '_tile_{}-{}.tif'

    with rasterio.open(tile) as inds:

        meta = inds.meta.copy()

        for window, transform in get_tiles(inds, N, N):

            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height

            Path(os.path.join(outputfolder, tile_name)).mkdir(exist_ok=True, parents=True)
            outpath = os.path.join(outputfolder, tile_name,
                                    output_filename.format(int(window.col_off), int(window.row_off)))
            arcpy.AddMessage("Processing: {}".format(outpath))

            array_val = inds.read(window=window)

            if array_val.shape[1] == array_val.shape[2] == 256:

                with rasterio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))

                patch_geom = get_tile_geom(outpath)
                patch_gdf = gdf[gdf.within(patch_geom)]

                if not patch_gdf.empty:
                    # move all subtiles that are inter-sect with the CUSP data to a separate folder, the imageries in this folder will be used
                    # to create training/validation data

                    patch_path = os.path.join(outputfolder,
                                                output_filename.format(int(window.col_off), int(window.row_off)))

                    shutil.copyfile(outpath, patch_path)

    # delete the non-intersect image patches
    shutil.rmtree(os.path.join(outputfolder, tile_name))



