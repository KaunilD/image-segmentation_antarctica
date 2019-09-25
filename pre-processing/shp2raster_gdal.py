import shapefile
import numpy as np
import glob
import cv2
import math
from osgeo import gdal, ogr, osr
import os



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
        image_fn = '{}.png'.format(image_id)
        print("Processing: {}".format(image_id))

        print("\tClipping {}.shp from {}".format(image_id, SHP_MASTER))
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

        print("Getting parameters from {}".format(image_id2image[image_id].get_path()))
        ds = gdal.Open(image_id2image[image_id].get_path())
        w = ds.RasterXSize
        h = ds.RasterYSize
        print(w, h)
        # get raster data
        print("\tRasterizing {}.shp to {}_mask.tif".format(image_id, image_id))
        os.system(
            'gdal_rasterize -burn 255 -ts {} {} {}.shp mask_{}.tif'
            .format(
                w, h,
                image_id, image_id
            )
        )
        print("\tWarping CRS to mask_{}.tif".format(image_id))
        ds = gdal.Warp(
            'epsg4326_mask_{}.tif'.format(image_id),
            'mask_{}.tif'.format(image_id),
            dstSRS='EPSG:4326',
            outputType=gdal.GDT_Int32
        )
        """
        gdal_translate .\\QB02_20120203205614_101001000EDC4600_12FEB03205614-M1BS-052813648020_01_P001_u16ns3031.tif
        -b 1 -b 2 -b 3
        rgb.tif
        -scale_1 173 852 0 65535
        -scale_2 222 1247 0 65535
        -scale_3 147 884 0 65535
        -co COMPRESS=DEFLATE
        -co PHOTOMETRIC=RGB
         """
        ds = None
