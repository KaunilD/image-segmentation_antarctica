import numpy as np
from sklearn.cluster import KMeans
import argparse
import cv2
import datetime
import time
import faiss
import glob


def get_filename_ext(string):
    fname = string.split("/")[-1]
    file_name, ext = fname.split(".")

    return file_name, ext


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False, help='Path to image file')
    ap.add_argument('-w', '--width', type=int, default=0,
        help='Width to resize image to in pixels')
    ap.add_argument('-s', '--color-space', type=str, default='bgr',
        help='Color space to use: BGR (default), HSV, Lab, YCrCb (YCC)')
    ap.add_argument('-c', '--channels', type=str, default='all',
        help='Channel indices to use for clustering, where 0 is the first channel,'
        + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" '
        + 'selects channels B and R. (default "all")')
    ap.add_argument('-n', '--num-clusters', type=int, default=3,
        help='Number of clusters for K-means clustering (default 3, min 2).')
    ap.add_argument('-o', '--output-file', action='store_true',
        help='Save output image (side-by-side comparison of original image and'
        + ' clustering result) to disk.')
    ap.add_argument('-f', '--output-format', type=str, default='png',
        help='File extension for output image (default png)')

    args = vars(ap.parse_args())
    return args


def main(args):

    start = time.time()

    image = cv2.imread(args['image'])

    # Resize image and make a copy of the original (resized) image.
    print("Resizing image.")
    if args['width'] > 0:
        height = int((args['width'] / image.shape[1]) * image.shape[0])
        image = cv2.resize(image, (args['width'], height),
            interpolation=cv2.INTER_AREA)
    orig = image.copy()

    print("Changing color spaces.")
    # Change image color space, if necessary.
    colorSpace = args['color_space'].lower()
    if colorSpace == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif colorSpace == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        colorSpace = 'bgr'  # set for file naming purposes

    # Keep only the selected channels for K-means clustering.
    print("Discarding channels.")
    if args['channels'] != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args['channels']:
            channelIndices.append(int(char))
        image = image[:,:,channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)

    # Flatten the 2D image array into an MxN feature vector, where M is
    # the number of pixels and N is the dimension (number of channels).
    print("Flattening channels.")
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    n_data, d = reshaped.shape

    reshaped = np.require(reshaped, dtype=np.float32, requirements=['C'])

    # Perform K-means clustering.
    print("Performing K-Means.")
    if args['num_clusters'] < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
    numClusters = max(2, args['num_clusters'])
    clus = faiss.Clustering(d, numClusters)
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # perform the training
    clus.train(reshaped, index)
    _, I = index.search(reshaped, 1)
    losses = faiss.vector_to_array(clus.obj)
    print('k-means loss evolution: {0}'.format(losses))

    # Reshape result back into a 2D array, where each element represents the
    # corresponding pixel's cluster index (0 to K - 1).
    print("Reshaping result into 2d array.")
    clustering = np.reshape(np.array(I, dtype=np.uint8),
        (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    print("Sorting cluster labels.")
    sortedLabels = sorted([n for n in range(numClusters)],
        key=lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    print("Setting pixel colors.")
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i

    print("Writing to file.")
    if args['output_file']:
        # Construct timestamped output filename and write image to disk.
        file_name, _ = get_filename_ext(args['image'])
        fileExtension = args['output_format']
        filename = file_name + '_mask.' + fileExtension
        cv2.imwrite(filename, kmeansImage)

    print("Time taken {}".format(time.time() - start))


if __name__ == "__main__":

    images_list = sorted(glob.glob(
        '../../data/pre-processed/dryvalleys/WV02/' + '*_3031.tif'))

    args = parse_args()
    for image in images_list:
        args['image'] = image
        main(args)
        print()
