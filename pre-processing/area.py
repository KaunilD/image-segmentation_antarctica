import shapefile
import numpy as np
import glob
import math
from osgeo import gdal, ogr, osr
import os

#os.environ["GDAL_DATA"] = 'C:\\Users\\dhruv\\.conda\\envs\\geology\\Library\\share\\gdal'

OUT_SUFFIX = '../data/pre-processed/dryvalleys/WV03'
GEOTIF_PATH = '../data/geotiffs/dryvalleys/WV03/*.tif'
SHP_PATH = '../data/shapefiles/dryvalleys/WV03'

SHP_MASTER = '../data/shapefiles/PGC_LIMA_VALID_3031-84.shp'

def main():
    poly="C:\\myshape.shp"
    shp = ogr.Open(poly)
    layer = shp.GetLayer()
    # For every polygon
    for index in xrange(len(allFID)):
        feature = layer.GetFeature(index)
        # get "FID" (Feature ID)
        FID = str(feature.GetFID())
        geometry = feature.GetGeometryRef()
        # get the area
        Area = geometry.GetArea()

if __name__=="__main__":
    main()
