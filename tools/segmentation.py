import os 
os.chdir('./tools')
import numpy as np 
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from features import compose_features, load_depth_estimator
import matplotlib.pyplot as plt
import argparse


def img2nodes(features, params) :
    """Converts the pixels into nodes for the graph embedding, can also group small blocks of pixels into a superpixel for computational efficiency. 

    Args:
        features (ndarray): the stacked image transformations chosen as features
        params (dict): the parameters of the pixels to nodes transformation, contains : 
                        - 'pos_weight' : the weight assigned to the positional arguments 
                        - 'supper_pixel' : the size of the super_pixel.

    Returns:
        s_pixels, s_pixels_indexes: the graph nodes
    """
    
    pixels = [[i*params['pos_weight'], j*params['pos_weight']]+list(features[i, j]) for i in range(features.shape[0]) for j in range(features.shape[1])]
    pixels = np.array(pixels)
    
    s_pixels = []
    i, j = 0, 0
    
    s_pixels_indexes = []
    
    while i<features.shape[0] :
        while j<features.shape[1] : 
            indexes = [(i+s_i)*(features.shape[0])+j+s_j for s_i in range(params['supper_pixel']) for s_j in range(params['supper_pixel'])]
            s_pixels.append(pixels[indexes].flatten())
            s_pixels_indexes.append(indexes)
            j += params['supper_pixel'] 
        j = 0
        i += params['supper_pixel']
      
    s_pixels = np.array(s_pixels) 
    return s_pixels , s_pixels_indexes
    
def manifold_embedding(nodes, params): 
    """construct the graph and project the data into a lower-dimensional space.

    Args:
        nodes (ndarray): the graph nodes
        params (dict): the parameters of the spectral embedding : 
                        - 'embedding_method' : manifold learning method => 'spectral' or 'isomap'
                        - 'n_components' : dimension of the projection space 
                        - 'n_neighbors' : number of neighbors used for the graph construction

    Returns:
        ndarray: projected data
    """

    if params['embedding_method']=='spectral' : 
        model = SpectralEmbedding(n_components=params['n_components'], n_neighbors=params['n_neighbors'])
    elif params['embedding_method']=='isomap' : 
        model = Isomap(n_components=params['n_components'], n_neighbors=params['n_neighbors'])
        
    X_proj = model.fit_transform(nodes) 
    
    return X_proj

def cluster(embedding, params) : 
    """Using the embedding given as input, this function will cluster the data into k clusters (image segments)

    Args:
        embedding (ndarray): projected pixels
        params (dict): the parameters of the clustering
                        - 'cluster_method' : clustering method => 'knn' or 'mixture'
                        - 'n_clusters' : number of clusters
    Returns:
        list: list of labels of each projected node
    """
    
    if params['cluster_method']=='knn' : 
        model = KMeans(n_clusters=params['n_clusters']).fit(embedding)
        labels = model.labels_
    elif params['cluster_method']=='mixture' : 
        model = GaussianMixture(n_components=params['n_clusters'], random_state=0).fit(embedding)
        labels = model.predict(embedding)
        
    return labels 

def segment(features, params={}) : 
    """main function that performs the segmentation by grouping the different blocks

    Args:
        features (ndarray): the stacked image transformations chosen as features
        params (dict): parameters of the segmentation, contains : 
                        - 'pos_weight' : the weight assigned to the positional arguments 
                        - 'supper_pixel' : the size of the super_pixel.
                        - 'embedding_method' : manifold learning method => 'spectral' or 'isomap'
                        - 'n_components' : dimension of the projection space 
                        - 'n_neighbors' : number of neighbors used for the graph construction
                        - 'cluster_method' : clustering method => 'knn' or 'mixture'
                        - 'n_clusters' : number of clusters

    Returns:
        labels_shaped: mask with the same shape of the image where each pixel refers to a cluster
        embedding : the projected data
        labels : 1D array of all the labels 
    """

    nodes, indexes = img2nodes(features, params)
    
    embedding = manifold_embedding(nodes, params)
    
    labels = cluster(embedding, params)
    
    labels_shaped = np.zeros(features.shape[0]*features.shape[1])
    for i in range(len(labels)) : 
        labels_shaped[indexes[i]] = labels[i]
        
    labels_shaped = labels_shaped.reshape(features.shape[0], features.shape[1])
    return labels_shaped, embedding, labels


if __name__ == '__main__' : 
    
    parser = argparse.ArgumentParser(description='Unsupervised Segmentation')
    parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
    parser.add_argument('--output', metavar='FILENAME',
                    help='output image file name', required=True)
    args = parser.parse_args()
    
    # path = '../images/101027.jpg'
    estimator = load_depth_estimator()
    
    #TODO: Modify the features weights to adapt to image domain
    features, t = compose_features(args.input, estimator, shape=(256, 256), weights=[0.8, 1.2, 0.8, 0.1])
    
    #TODO: Modify the spectral embedding parameters to adatpt to image domain
    params = {'supper_pixel':2, 'embedding_method':'spectral', 'n_components':10, 'n_neighbors':50, 'cluster_method':'knn', 'n_clusters':7, 'pos_weight':0.4}
    labels_shaped, embedding, labels = segment(features, params=params)
    
    fig, axes = plt.subplots(1, len(t)+1, figsize=(16, 4))

    for i in range(len(t)) : 
        
        axes[i].imshow(t[i])
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i}')
        
    axes[-1].imshow(labels_shaped)
    axes[-1].axis('off')
    axes[-1].set_title(f'Segementation')
    
    plt.savefig(args.output)
    # plt.show()