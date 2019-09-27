import shapefile
import numpy as np
import glob
import cv2
import math
from osgeo import gdal, ogr, osr
import os

os.environ["GDAL_DATA"] = 'C:\\Users\\dhruv\\.conda\\envs\\geology\\Library\\share\\gdal'

GEOTIF_PATH = '../data/gettiffs/dryvalleys/*.tif'
SHP_PATH = '../data/shapefiles/dryvalleys/image_id*.shp'

SHP_MASTER = '../data/shapefiles/PGC_LIMA_VALID_4326-84.shp'

class Image(object):
    def __init__(self, path):
        self._path = path
        self._id = path.split("_")[2]
        self._polygons = []
        self._bbox = None
        self._image = None
        self._map = None

    def get_polygons(self):
        return self._polygons

    def get_poly(self, idx):
        return self._polygons[idx]

    def add_poly(self, poly):
        self.get_polygons().append(poly)
        return None

    def get_id(self):
        return self._id

    def get_path(self):
        return self._path

    def get_bbox(self):
        return self._bbox

    def set_image(self, data):
        self._image = data

    def set_bbox(self, data):
        self._bbox = data

if __name__ == "__main__":
    driver = ogr.GetDriverByName("ESRI Shapefile")
    image_id2image = {}
    image_id2polygons = {}

    for path in glob.glob(GEOTIF_PATH):
        image = Image(path)
        image_id2image[image.get_id()] = image

    for path in glob.glob(SHP_PATH):
        sf = shapefile.Reader(path)
        shapeRecs = sf.shapeRecords()
        for shapeRec in shapeRecs:
            image_id = shapeRec.record[0]
            shape = shapeRec.shape.bbox
            if image_id in image_id2image:
                image_id2image[image_id].set_bbox(shape)

    shp_master_ds = driver.Open(SHP_MASTER, 1)
    layer = shp_master_ds.GetLayer()
    shp_master_srs = layer.GetSpatialRef()

    for image_id in image_id2image.keys():
        print("Processing: {}".format(image_id))
        print()

        print("Clipping {}.shp from {}".format(image_id, SHP_MASTER))
        bbox = image_id2image[image_id].get_bbox()
        os.system(
            'ogr2ogr -f "ESRI Shapefile" {}.shp {} -clipsrc {}'
            .format(
            image_id,
            SHP_MASTER,
            '../data/shapefiles/dryvalleys/image_id_{}.shp'.format(image_id)
            )
        )

        # Get a Layer's Extent
        inShapefile = '{}.shp'.format(image_id)
        inDriver = ogr.GetDriverByName("ESRI Shapefile")
        inDataSource = inDriver.Open(inShapefile, 0)
        inLayer = inDataSource.GetLayer()
        extent = inLayer.GetExtent()
        print(extent)

        print("Warping {}_rgb.png".format(image_id))
        gdal.Warp(
            "{}_4326.tif".format(image_id),
            image_id2image[image_id].get_path(),
            options=gdal.WarpOptions(
                dstSRS='EPSG:4326'
            )
        )
        tif_ds = gdal.Translate(
            "{}_4326_cropped.png".format(image_id), "{}_4326.tif".format(image_id),
            format='PNG', outputType=gdal.GDT_Byte,
            projWin = [extent[0], extent[3], extent[1], extent[2]],
            projWinSRS = 'EPSG:4326',
            scaleParams=[[173, 852], [222, 1247], [147, 884]],
        )

        w = tif_ds.RasterXSize
        h = tif_ds.RasterYSize

        print("Raster WIDTH = {} HEIGHT = {}".format(w, h))
        print()
        # get raster data
        print("Rasterizing {}.shp to {}_mask.tif".format(image_id, image_id))
        ds = gdal.Rasterize(
            '{}_mask_4326.tif'.format(image_id),
            '{}.shp'.format(image_id),

            options=gdal.RasterizeOptions(
                burnValues=255,
                allTouched=True,
                width = w,
                height = h,
                outputSRS = 'EPSG:4326',
                outputType = gdal.GDT_Byte,
                outputBounds = image_id2image[image_id].get_bbox()
            )
        )

        print()
