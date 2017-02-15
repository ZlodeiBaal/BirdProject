import cv2
import time
import caffe
import numpy as np
from scipy import misc

net = caffe.Net('deploy.prototxt', 'SQ.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1, 3,  227, 227)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
mu = np.array([128.0, 128.0, 128.0])
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)
#transformer.set_channel_swap('data', (1,2,0))

video_capture = cv2.VideoCapture(0)
video_capture.set(3,1280)
video_capture.set(4,720)
video_capture.set(10, 0.6)
ret, frame_old = video_capture.read()
i=0
j=0
k=0
while True:
    time.sleep(0.1)
    ret, frame = video_capture.read()
    diffimg = cv2.absdiff(frame, frame_old)
    d_s = cv2.sumElems(diffimg)
    d = (d_s[0]+d_s[1]+d_s[2])/(1280*720)
    frame_old=frame
    print d
    if i>30:
        if (d>20):
            #frame = frame[:, :, ::-1]
            frame = frame[:, :, [2, 1, 0]]
	    transformed_image = transformer.preprocess('data', frame)
            net.blobs['data'].data[0] = transformed_image
            net.forward()
            if (net.blobs['pool10'].data[0].argmax()!=0):
                #frame = frame[:, :, ::1]
                #cv2.imwrite("base/"+str(j)+"_"+str(net.blobs['pool10_Q'].data[0].argmax())+".jpg", frame)
                misc.imsave("base/"+str(j)+"_"+str(net.blobs['pool10_Q'].data[0].argmax())+".jpg",frame)
                j=j+1
            else:
                #cv2.imwrite("base_d/"+str(k)+".jpg", frame)
                misc.imsave("base_d/"+str(k)+".jpg",frame)
                k=k+1
    else:
        i=i+1
