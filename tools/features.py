from skimage.color import rgb2gray,rgb2hsv
import cv2 
from transformers import pipeline
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt


def hsv_transform(img) : 
    """_summary_

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (255*rgb2hsv(img)).astype(int)

def load_depth_estimator() : 
    """_summary_

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    
    return estimator

def depth_transform(img_path, estimator, shape) : 
    """_summary_

    Args:
        img_path (_type_): _description_
        estimator (_type_): _description_

    Returns:
        _type_: _description_
    """
    d_img = np.array(estimator(img_path)['depth'])
    d_img = cv2.resize(d_img, shape)
    return d_img

def savola_transform(img) : 
    """_summary_

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    gray_image = rgb2gray(img)
    threshold = filters.threshold_sauvola(gray_image)
    binarized_image = (gray_image > threshold)*1
    
    return (255*binarized_image).astype(int)

def compose_features(img_path, estimator, shape=(256, 256), weights=[1, 1, 1, 1])  :
    """_summary_

    Args:
        img (_type_): _description_
        estimator (_type_): _description_
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, shape)
    
    h_img = hsv_transform(img)
    d_img = depth_transform(img_path, estimator, shape)
    b_img = savola_transform(img)
    
    # print(img.shape, h_img.shape)
    features  = np.zeros((shape[0], shape[1], img.shape[2]+2+h_img.shape[2]))
    features[:, :, :3] = img*weights[0] 
    features[:, :, 3:6] = h_img*weights[1] 
    features[:, :, 6] = d_img*weights[2] 
    features[:, :, 7] = b_img*weights[3] 
    
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(int)
    
    return features, (img, d_img, h_img, b_img)


if __name__=='__main__' : 
    
    path = './images/101027.jpg'
    estimator = load_depth_estimator()
    features, t = compose_features(path, estimator, shape=(256, 256), weights=[1, 1, 1, 1])
    
    fig, axes = plt.subplots(1, len(t), figsize=(12, 4))

    for i in range(len(t)) : 
        
        axes[i].imshow(t[i])
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i}')
    
    plt.savefig('images/features.png')
    # plt.show()