import caffe
import numpy as np
from scipy import misc
import cv2

#Miminalisticc test
caffe_root = '/media/anton/WorkAndStuff/CAFFE/caffenew0217/'
net = caffe.Net('deploy.prototxt', 'SQ.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1, 3,  227, 227)
image = misc.imread('/media/anton/Bazes/BIRD/FinalBase/1227.jpg')
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

mu = np.array([128.0, 128.0, 128.0])
#transformer.set_channel_swap('data', (1,2,0))
transformer.set_transpose('data', (2, 0, 1))

transformer.set_mean('data', mu)

transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[0] = transformed_image
net.forward()
print net.blobs['pool10'].data[0]
print net.blobs['pool10_Q'].data[0].argmax()