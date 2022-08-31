
import arcpy
import rasterio
import geopandas as gpd
from itertools import product
import shutil
import os
from pathlib import Path
import shapely
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import torch
from rasterio.merge import merge
from rasterio import features
import pandas as pd
from shapely.geometry import shape
from rasterstats import zonal_stats

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

def logits2conf(img):

    Xmax = np.amax(img)
    Xmin = np.amin(img)

    newX = (img - Xmin) / (Xmax - Xmin)

    return newX

def mosaic_list(imglist, outpath):

    raster_to_mosiac = []

    for conf in imglist:
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

    with rasterio.open(outpath, "w", **output_meta) as m:
        m.write(mosaic)

#############################################################
# Input parameters
# proj_path = arcpy.GetParameterAsText(0)
# data_dir = arcpy.GetParameterAsText(1)
# geo_data = arcpy.GetParameterAsText(2)
# buffer_unit = int(arcpy.GetParameterAsText(3)) * 0.00001

proj_path = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS"
data_dir = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\data"
geo_data = r"C:\\Users\mlv\Documents\projects\ShorelineArmoring_ArcGIS\data\VA_CUSP_selected_wgs84.shp"
buffer_unit = int(3) * 0.00001
#############################################################

#############################################################
# Step 1. Image cropping
#############################################################
cropfolder_temp = os.path.join(proj_path, 'crop_temp')
Path(cropfolder_temp).mkdir(exist_ok=True, parents=True)

N = 256
gdf = gpd.read_file(geo_data)

all_tiles = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('tif')]

cut = False
while cut:
    cut = False
    for tile in all_tiles:

        tile_name = os.path.basename(tile).split('.')[0]
        output_filename = tile_name + '_tile_{}-{}.tif'

        with rasterio.open(tile) as inds:

            meta = inds.meta.copy()

            for window, transform in get_tiles(inds, N, N):

                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height

                Path(os.path.join(cropfolder_temp, tile_name)).mkdir(exist_ok=True, parents=True)
                outpath = os.path.join(cropfolder_temp, tile_name,
                                        output_filename.format(int(window.col_off), int(window.row_off)))
                arcpy.AddMessage("Processing: {}".format(outpath))

                array_val = inds.read(window=window)

                if array_val.shape[1] == array_val.shape[2] == 256:

                    with rasterio.open(outpath, 'w', **meta) as outds:
                        outds.write(inds.read(window=window))

                    patch_geom = get_tile_geom(outpath)
                    patch_gdf = gdf[gdf.within(patch_geom)]

                    if not patch_gdf.empty:

                        patch_path = os.path.join(cropfolder_temp,
                                                    output_filename.format(int(window.col_off), int(window.row_off)))

                        shutil.copyfile(outpath, patch_path)

        # delete the non-intersect image patches
        shutil.rmtree(os.path.join(cropfolder_temp, tile_name))

#############################################################
# Step 2. Image prediction
image_path = proj_path
#############################################################

model_path = os.path.join(proj_path, "weights","model.pth")
predict_temp = os.path.join(proj_path, "predict_temp")
Path(predict_temp).mkdir(exist_ok=True, parents=True)

# get all image patches in cropfolder_temp
allfiles = [file for file in os.listdir(cropfolder_temp) if file.endswith(".tif")]
arcpy.AddMessage("Processing: {}".format(allfiles))

# load model architecture
model = smp.Unet('resnet18',
                encoder_weights="imagenet",
                in_channels=4,
                classes=4)

# load weights
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

transform_test = transforms.Compose([transforms.ToTensor()])
impute_nan = np.tile([0, 0, 0, 0], (256,256,1))

with torch.no_grad():
    model.eval()

    for roi in allfiles:

        arcpy.AddMessage("Working on {}".format(roi))

        roi_file = os.path.join(cropfolder_temp, roi)

        output_image = os.path.join(predict_temp,os.path.basename(roi_file).split('.tif')[0] + '_unet.tif')
        output_conf = os.path.join(predict_temp, os.path.basename(roi_file).split('.tif')[0] + '_unet_conf.tif')

        # Read metadata of the initial image
        with rasterio.open(roi_file, mode ='r') as src:
            tags = src.tags().copy()
            meta = src.meta
            image = src.read()
            image = np.moveaxis(image, (0, 1, 2), (2, 0, 1)).astype('float32')
            dtype = src.read(1).dtype

        # Update meta to reflect the number of layers
        meta.update(count = 1)

        # Write prediction class
        with rasterio.open(output_image, 'w', **meta) as dst:

            nan_mask = np.isnan(image)
            image[nan_mask] = impute_nan[nan_mask]

            image = transform_test(image)
            logits = model(image.unsqueeze(0))

            conf = np.amax(logits.numpy(), axis=1).squeeze()
            conf = logits2conf(conf).squeeze()

            probs = logits.softmax(dim=1).numpy().argmax(1).squeeze()

            dst.write_band(1, probs.astype(dtype).copy()) # In order to be in the same dtype
            # dst.update_tags(**tags)

        # Write confidence (update the export file data type)
        meta.update(dtype='float32')
        with rasterio.open(output_conf, 'w', **meta) as dst:
            dst.write_band(1, conf)

#############################################################
# Create a mosaic confidence layer
output_path = os.path.join(proj_path, 'mosaic_conf.tif')

allconfs = [os.path.join(predict_temp, f) for f in os.listdir(predict_temp) if f.endswith('conf.tif')]
mosaic_list(allconfs, output_path)
#############################################################

#############################################################
# Step 3. Post processing
# This script is used to burn the coastal line features into the shortline structure detection result
# to improve the overall accuracy
# output directory
shoreline_masks = os.path.join(proj_path, 'predict_class')
Path(shoreline_masks).mkdir(exist_ok=True, parents=True)
predicted_shp = os.path.join(proj_path, 'test_output.geojson')
#############################################################

gdf = gpd.read_file(geo_data)
model_masks_paths = [os.path.join(predict_temp, file) for file in os.listdir(predict_temp) if file.endswith("unet.tif")]


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
rdf.to_file(predicted_shp, driver='GeoJSON')

#############################################################
# Step 4. Zonal Statistics
stats = zonal_stats(predicted_shp, output_path, stats=['min', 'max', 'median', 'sum', 'count', 'mean'])
stats_df = pd.DataFrame(stats)
poly_df = gpd.read_file(predicted_shp)
pd = poly_df.join(stats_df)

pd.to_file(os.path.join(proj_path, 'vector_results.shp'))
os.remove(predicted_shp)






