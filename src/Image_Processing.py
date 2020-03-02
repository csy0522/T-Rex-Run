# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:29:35 2020

@author: CSY
"""


from PIL import Image
import base64
from io import BytesIO
import cv2
import numpy as np

'''
Hyper Parameters
'''
# Canny
canny1 = 100
canny2 = 100

# New height and width for cropping
cropped_y1 = 50
cropped_y2 = 0
cropped_x1 = 0
cropped_x2 = 400



'''
Parameters:
    img: the image that is going to be normalized
What this function does:
    Normalizes all the pixel values by dividing
    each pixel by the maximum value of the pixels
'''
def normalize(img):
    return img/float(img.max())


'''
Parameters:
    img: The image before cropping
    y1 & y2: the new height of the image (from y1 to y2)
    x1 & x2: the new width of the image (from x1 to x2)
What this function does:
    Crops out a certain portion of an image
    with the given parameters
'''
def crop(img,y1,y2,x1,x2):
    return img[y1:y2,x1:x2]


'''
Parameters
    t: type of image processing,
        either "gray" or "canny" for now, more coming up in the future
    img: the image that will be processed.
What this function does:
    Formats the image to the image that the RL model will be using.
    The proces:
        3d -> 2d -> cropping -> resizing -> normalizing
'''
def img_processing(t,r_row,r_column,img):
    processed = None
    if t == "gray":
        processed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    elif t == "canny":
        processed = cv2.Canny(img,canny1,canny2)
    cropped_y2 = processed.shape[0]
    cropped = crop(processed,
                   cropped_y1,cropped_y2,cropped_x1,cropped_x2)
    resized = cv2.resize(cropped,(r_row,r_column)).astype("float32")
    final = normalize(resized)
    return final


'''
Parameters:
    screen: The binary code from a screenshot of the image using selenium webdriver
What this function does:
    It transfers the binary code of a screenshot to an array image
'''
def screenshot(screen):
    img = Image.open(BytesIO(base64.b64decode(screen)))
    return np.array(img)


'''
Parameters:
    img: an image in array
What this function does:
    It saves screenshot as jpg
'''
def save_as_jpg(img):
    return cv2.imwrite("screenshot.jpg", img)



'''
Parameters:
    imgs: a list of images
    ind: index of stacking
What this function does:
    Stacks a given image to lay layers at ind index
'''
def stack_images(imgs, ind):
    if ind > len(imgs[0].shape):
        ind = len(imgs[0].shape)
    stacks = []
    for img in imgs:
        stacks.append(img)
    return np.stack(stacks,ind)
    

    
'''
Parameters:
    dim: the dimention that the image will be converted to
    img: an image that could be 2d, 3d or 4d.
What this function does:
    Reshapes the image to either 3d or 4d image
''' 
def reshape_to(dim,img):
    d = len(img.shape)
    if dim == 1:
        if d == 2:
            return img.reshape(img.shape[0])
    if dim == 3:
        if d == 3:
            return img
        elif d == 2:
            return img.reshape(img.shape[0],img.shape[1],1)
        elif d == 4:
            return img.reshape(img.shape[0],img.shape[1],img.shape[2])
        else:
            return "Dimention of the image must be either 2 or 4"
    elif dim == 4:
        if d == 2:
            return img.reshape(1,img.shape[0],img.shape[1],1)
        elif d == 3:
            return img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        elif d == 5:
            return img.reshape(img.shape[0],img.shape[2],img.shape[3],img.shape[4])
        else:
            return "Dimention of the image must be either 2 or 3"
        
    
    
    
'''
Parameters:
    new_img: the new image that is to be inserted to the stack
    stack: the stack of images which the first image will be replaced to the new_img
What this function does:
    Works as a FIFO list of images. Each individual image is 3d, and the stack is 4d
'''
def fifo_images(new_img,stack):
    new_img_4d = reshape_to(4,new_img)
    return np.append(stack[:,:,:,1:],new_img_4d,axis=3)


    
'''
Parameters:
    a: an array
What this function does:
    Converts an array to a matrix
'''
def a2m(a):
    return np.matrix(a)
    

