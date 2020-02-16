import shapefile
import numpy as np
import glob
import math
from osgeo import gdal, ogr, osr
import os

#os.environ["GDAL_DATA"] = 'C:\\Users\\dhruv\\.conda\\envs\\geology\\Library\\share\\gdal'

OUT_SUFFIX = '../data/pre-processed/dryvalleys/WV03'
GEOTIF_PATH = '../data/gettiffs/dryvalleys/*.tif'
SHP_PATH = '../data/shapefiles/dryvalleys/WV03'

SHP_MASTER = '../data/shapefiles/PGC_LIMA_VALID_3031-84.shp'

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

    shp_master_ds = driver.Open(SHP_MASTER, 1)
    layer = shp_master_ds.GetLayer()
    shp_master_srs = layer.GetSpatialRef()

    for image_id in image_id2image.keys():
        print("Processing: {}".format(image_id))
        print()

        print("Warping {}/{}_4326.tif".format(OUT_SUFFIX, image_id))
        tif_ds = gdal.Open(
            image_id2image[image_id].get_path()
        )

        tif_ds = gdal.Translate(
            "{}/{}_3031.tif".format(OUT_SUFFIX, image_id), image_id2image[image_id].get_path(),
            format='GTiff', outputType=gdal.GDT_Byte,
            bandList=[8, 4, 1],
            scaleParams=[
                [tif_ds.GetRasterBand(8).GetStatistics(0, 1)[0], tif_ds.GetRasterBand(8).GetStatistics(0, 1)[1]],
                [tif_ds.GetRasterBand(4).GetStatistics(0, 1)[0], tif_ds.GetRasterBand(4).GetStatistics(0, 1)[1]],
                [tif_ds.GetRasterBand(1).GetStatistics(0, 1)[0], tif_ds.GetRasterBand(1).GetStatistics(0, 1)[1]],
            ],
        )

        gt = tif_ds.GetGeoTransform()
        w = tif_ds.RasterXSize
        h = tif_ds.RasterYSize

        ext_ = get_extent(gt, w, h)
        print(ext_)
        # get raster data
        print("Rasterizing {}/{}.shp to {}/{}_3031_mask.png".format(OUT_SUFFIX, image_id, OUT_SUFFIX, image_id))
        print("\tWIDTH = {} HEIGHT = {}".format(w, h))
        ds = gdal.Rasterize(
            '{}/{}_3031_mask.tif'.format(OUT_SUFFIX, image_id),
            '{}'.format(SHP_MASTER),

            options=gdal.RasterizeOptions(
                burnValues=255,
                allTouched=True,
                width = w,
                height = h,
                outputSRS = 'EPSG:3031',
                outputType = gdal.GDT_Byte,
                format='GTiff',
                outputBounds = [ext_[0][0], ext_[0][1], ext_[2][0], ext_[2][1]]
            )
        )

        print()
