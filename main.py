from platform import python_version
import tensorflow
import keras
import cvlib as cv
import cv2

print('Python version: {}'.format(python_version()))
print('cvlib version: {}'.format(cv.__version__))
print('OpenCV version: {}'.format(cv2.__version__))
print('Tensorflow version: {}'.format(tensorflow.__version__))
print('Keras version: {}'.format(keras.__version__))

import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox

im = io.imread('http://www.samacharnama.com/wp-content/uploads/2019/06/third-party-insurance..687.png')
plt.imshow(im)
plt.show(None)
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()
print('Number of cars in the image is ' + str(label.count('car')))


