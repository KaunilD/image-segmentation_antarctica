import shapefile
import numpy as np
import glob
import cv2
import math
from osgeo import gdal



GEOTIF_PATH = '../data/gettiffs/dryvalleys/*.tif'
SHP_PATH = '../data/shapefiles/dryvalleys/*.shp'
vmin = 0 # minimum value in your data (will be black in the output)
vmax = 255 # minimum value in your data (will be white in the output)


def lonlat2xy(lonlat, map_shape):
    w, h = map_shape[:2]
    lon, lat = lonlat
    x = (lon+180)*(w/360)

    latRad = lat*math.pi/180
    mercN = math.log(math.tan((math.pi/4)+(latRad/2)))
    y     = (h/2)-(w*mercN/(2*math.pi))
    return int(x), int(y)

class Image(object):
    def __init__(self, path):
        self._path = path
        self._id = path.split("_")[2]
        self._polygons = []
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

    def set_image(self, data):
        self._image = data

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
            shape = shapeRec.shape.points
            if image_id in image_id2image:
                image_id2image[image_id].add_poly(shape)

    for image_id in image_id2image.keys():
        image_fn = '{}.png'.format(image_id)
        print(image_id)
        ds = gdal.Translate(image_fn, image_id2image[image_id].get_path(), format='PNG', outputType=gdal.GDT_Byte, scaleParams=[[]])

        r = ds.GetRasterBand(3)
        g = ds.GetRasterBand(2)
        b = ds.GetRasterBand(1)

        data = np.dstack([r.ReadAsArray(), g.ReadAsArray(), b.ReadAsArray()])
        image_id2image[image_id].set_image(data)

        map = np.zeros(data.shape)

        polygons = image_id2image[image_id].get_polygons()
        for polygon in polygons:
            point1 = polygon[0]

            for point2 in polygon[1:]:
                x1, y1 = lonlat2xy(point1, map.shape)
                x2, y2 = lonlat2xy(point2, map.shape)
                print(x1, y1, x2, y2)
                cv2.line(map, (x1, y1), (x2, y2), (255,255,255), 1)
                point1 = point2
            print()

        cv2.imwrite('map_{}'.format(image_fn), map)

        ds = None

    # sf = shapefile.Reader(SHP_PATH)
