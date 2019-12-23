import cv2
import numpy as np

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width).
    '''
    #My Edit
    im = np.copy(input_image)
    #Resize image
    im = cv2.resize(im, (456,256))  #(W, H) AND Not (H,W) !!!
    #print(im.shape) #Prints (H, W, C)
    im = im.transpose((2,0,1))   #Gives (C, H, W)
    #Add another dimension to it
    im = im.reshape(1, 3, 256, 456)
    return im


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
        #My Edit
    im = np.copy(input_image)
    #Resize image
    im = cv2.resize(im, (1280,768))  #(W, H) AND Not (H,W) !!!
    #print(im.shape) #Prints (H, W, C)
    im = im.transpose((2,0,1))   #Gives (C, H, W)
    #Add another dimension to it
    im = im.reshape(1, 3, 768, 1280)
    return im

def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
        #My Edit
    im = np.copy(input_image)
    #Resize image
    im = cv2.resize(im, (72,72))  #(W, H) AND Not (H,W) !!!
    #print(im.shape) #Prints (H, W, C)
    im = im.transpose((2,0,1))   #Gives (C, H, W)
    #Add another dimension to it
    im = im.reshape(1, 3, 72, 72)
    return im
