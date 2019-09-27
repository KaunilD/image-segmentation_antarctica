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

vmin = 0 # minimum value in your data (will be black in the output)
vmax = 255 # minimum value in your data (will be white in the output)


def get_extent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def lonlat2xy(lonlat, map_shape, bbox):
    w, h = map_shape[:2]

    lon, lat = lonlat
    x, y = 0, 0

    x = ((lon - bbox[0])/(bbox[2]-bbox[0]))*w
    y = ((lat - bbox[1])/(bbox[3]-bbox[1]))*h
    return int(x), int(y)

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

        tif_ds = gdal.Open(
            image_id2image[image_id].get_path()
        )

        tif_srs = osr.SpatialReference()
        tif_srs.ImportFromWkt(tif_ds.GetProjection())



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
        extent = reproject_coords([[extent[0], extent[3]], [extent[1], extent[2]]], shp_master_srs, tif_srs)
        print(extent)

        print("Warping {}_rgb.png".format(image_id))
        tif_ds = gdal.Translate(
            "{}_rgb.png".format(image_id), image_id2image[image_id].get_path(),
            format='PNG', outputType=gdal.GDT_Byte,
            projWin = [extent[0][0], extent[0][1], extent[1][0], extent[1][1]],
            projWinSRS = 'EPSG:3031',
            scaleParams=[[173, 852], [222, 1247], [147, 884]],
        )

        tif_ds = gdal.Open(
            '{}_rgb.tif'.format(image_id),
        )
        w = tif_ds.RasterXSize
        h = tif_ds.RasterYSize

        """
        gt = tif_ds.GetGeoTransform()
        ext = get_extent(gt, w, h)

        tif_srs = osr.SpatialReference()
        tif_srs.ImportFromWkt(tif_ds.GetProjection())


        ext = reproject_coords(ext, shp_master_srs, tif_srs)
        """

        print("Raster WIDTH = {} HEIGHT = {}".format(w, h))
        print()
        # get raster data
        print("Rasterizing {}.shp to {}_mask.tif".format(image_id, image_id))
        ds = gdal.Rasterize(
            '{}_mask.tif'.format(image_id),
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
