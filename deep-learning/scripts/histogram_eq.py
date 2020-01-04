import cv2
import numpy as np


def main():
    img_path = "C:/Users/dhruv/Development/git/independent-study/data/pre-processed/dryvalleys/WV02/1_/103001001E0B4100_3031.tif"
    img = cv2.imread(img_path)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite("ing_equ.png", img_output )

if __name__=="__main__":
    main()
