import caffe
import numpy as np
from scipy import misc
import cv2

#Miminalisticc test
caffe_root = '/media/anton/WorkAndStuff/CAFFE/caffenew0217/'
net = caffe.Net('deploy.prototxt', 'SQ_NEW_iter_500.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1, 3,  227, 227)
#image = misc.imread('/media/anton/WorkAndStuff/OpenProject/BirdProject/SqNet/1___2.jpg')
for i in range(0,1827):
    s = '/home/anton/LocalDrive/RaspberryPi/base_d/' + str(i)+'.jpg'
    image = misc.imread(s)
    image = misc.imresize(image, [227, 227])
    #image = misc.imread('/media/anton/Bazes/BIRD/FinalBase/969.jpg')
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    mu = np.array([128.0, 128.0, 128.0])
    #transformer.set_channel_swap('data', (1,2,0))
    transformer.set_transpose('data', (2, 0, 1))

    transformer.set_mean('data', mu)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[0] = transformed_image
    net.forward()
    if (net.blobs['pool10'].data[0].argmax()!=0):
        print s