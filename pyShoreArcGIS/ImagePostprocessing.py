
# This script is used to burn the coastal line features into the shortline structure detection result
# to improve the overall accuracy

import rasterio
import os
import geopandas as gpd
from rasterio import features
import shapely
import arcpy
import pandas as pd
from shapely.geometry import shape
from pathlib import Path
import numpy as np

# Inputs
#
# coastal_file = arcpy.GetParameterAsText(0) # Coastal shapefile
# buffer_unit = int(arcpy.GetParameterAsText(1)) * 0.00001 # buffer value
# predicted_dir = arcpy.GetParameterAsText(2) # Predicted mask path
#
# # output directory
# shoreline_masks = arcpy.GetParameterAsText(3)
# predicted_shp = arcpy.GetParameterAsText(4)

coastal_file = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\data\VA_CUSP_wgs84.shp"

buffer_unit = int(1) * 0.00001
predicted_dir = r"C:\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\outputmask"

# output directory
shoreline_masks = r"C:\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\finaloutput"
predicted_shp = r"C:\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\test_output.shp"
Path(shoreline_masks).mkdir(exist_ok=True, parents=True)

gdf = gpd.read_file(coastal_file)
model_masks_paths = [os.path.join(predicted_dir, file) for file in os.listdir(predicted_dir) if file.endswith("unet.tif")]

def get_tile_geom(tile_tif):

    rds = rasterio.open(tile_tif)
    minx, miny, maxx, maxy = rds.bounds
    geometry = shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)

    return geometry


def vectorize_tiff(raster_file):
    with rasterio.open(raster_file) as src:
        image = src.read(1)

    # Remove backgrounds
    mask = image != 0

    # vectorize the image and export as a dataframe
    results = (
        {'class_value': v, 'geometry': s}
        for i, (s, v)
        in enumerate(features.shapes(image, mask=mask, transform=src.transform)))

    df = pd.DataFrame(results)

    return df


def add_class4_name(x):
    if int(x) == 0:
        name = 'bulkhead'

    if int(x) == 1:
        name = 'riprap'

    if int(x) == 2:
        name = 'groin'

    elif int(x) == 3:
        name = 'breakwater'

    return name




for tile_path in model_masks_paths:

    print("------------------------------------------------------")

    print("Processing tile {}: ".format(tile_path))

    # select vectors that are within the tile
    tif_geom = get_tile_geom(tile_path)

    # get the geometry of intersected shapes
    sub_gdf = gdf[gdf.intersects(tif_geom)]

    # Check the vectors that overlayed with the selected tile
    if not sub_gdf.empty:
        sub_gdf['geometry'] = sub_gdf['geometry'].buffer(buffer_unit)
        shape_polys = sub_gdf["geometry"].to_list()

        output_filename = os.path.basename(tile_path).split('.')[0] + '_shoreline_mask.tif'

        with rasterio.open(tile_path) as src:
            meta = src.meta.copy()
            meta['count'] = 1

            mask_array = src.read(1) # mask_array is the predicted multiclass array

            image = features.rasterize(
                ((shape_poly, 1) for shape_poly in shape_polys),
                out_shape=src.shape,
                transform=src.transform)

            filtered_mask = mask_array * image # The image is an array of 0/1 indicating the mask

            with rasterio.open(os.path.join(shoreline_masks, output_filename), 'w', **meta) as dst:
                dst.write(filtered_mask, indexes=1)


# merge the raster files into one
predict_masks_paths = [os.path.join(shoreline_masks, file) for file in os.listdir(shoreline_masks) if file.split(".")[-1]=="tif"]

dataframelist = list()
for file in predict_masks_paths:

    dataframelist.append(vectorize_tiff(file))


rdf = gpd.GeoDataFrame(pd.concat(dataframelist, ignore_index=True))
rdf['class_name'] = rdf['class_value'].apply(lambda x: add_class4_name(x))
rdf['geometry'] = rdf["geometry"].apply(lambda x: shape(x))
rdf.set_crs('epsg:4326')
rdf.to_file(predicted_shp)








