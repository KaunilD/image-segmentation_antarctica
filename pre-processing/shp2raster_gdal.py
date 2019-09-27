import shapefile
import numpy as np
import glob
import cv2
import math
from osgeo import gdal, ogr, osr
import os

os.environ["GDAL_DATA"] = 'C:\\Users\\dhruv\\.conda\\envs\\geology\\Library\\share\\gdal'


GEOTIF_PATH = '../data/gettiffs/dryvalleys/*.tif'
SHP_PATH = '../data/shapefiles/dryvalleys/*.shp'

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

    for image_id in image_id2image.keys():
        print("Processing: {}".format(image_id))


        print("Generating RGB mask {}_rgb.png".format(image_id))
        ds = gdal.Translate(
            "{}_rgb.png".format(image_id), image_id2image[image_id].get_path(),
            format='PNG', outputType=gdal.GDT_Byte,
            scaleParams=[[173, 852], [222, 1247], [147, 884]],
        )
        gt = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        ext = get_extent(gt, w, h)

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())

        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(4326)

        ext = reproject_coords(ext, src_srs, tgt_srs)

        print("Raster WIDTH = {} HEIGHT = {}".format(w, h))
        print(ext)
        print()

        print("Clipping {}.shp from {}".format(image_id, SHP_MASTER))
        bbox = image_id2image[image_id].get_bbox()
        os.system(
            'ogr2ogr -f "ESRI Shapefile" {}.shp {} -clipsrc {} {} {} {}'
            .format(
            image_id,
            SHP_MASTER,
            bbox[0], bbox[1],
            bbox[2], bbox[3]
            )
        )
        print()
        # get raster data
        print("Rasterizing {}.shp to {}_mask.tif".format(image_id, image_id))
        ds = gdal.Rasterize(
            '{}_mask.tif'.format(image_id),
            '{}.shp'.format(image_id),
            options=gdal.RasterizeOptions(
                burnValues=255,
                width = w,
                height = h,
                outputSRS = 'EPSG:4326',
                outputType = gdal.GDT_Byte
            )
        )
        print()
