import os 
os.chdir('./tools')
import numpy as np 
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from features import compose_features, load_depth_estimator
import matplotlib.pyplot as plt


def img2nodes(features, params) :
    """_summary_

    Args:
        features (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        nodes (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """

    if params['embedding_method']=='spectral' : 
        model = SpectralEmbedding(n_components=params['n_components'], n_neighbors=params['n_neighbors'])
    elif params['embedding_method']=='isomap' : 
        model = Isomap(n_components=params['n_components'], n_neighbors=params['n_neighbors'])
        
    X_proj = model.fit_transform(nodes) 
    
    return X_proj

def cluster(embedding, params) : 
    """_summary_

    Args:
        embedding (_type_): _description_
        params (_type_): _description_
    """
    
    if params['cluster_method']=='knn' : 
        model = KMeans(n_clusters=params['n_clusters']).fit(embedding)
        labels = model.labels_
    elif params['cluster_method']=='mixture' : 
        model = GaussianMixture(n_components=params['n_clusters'], random_state=0).fit(embedding)
        labels = model.predict(embedding)
        
    return labels 

def segment(features, params={}) : 
    """_summary_

    Args:
        features (_type_): _description_
        params (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
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
    
    path = '../images/101027.jpg'
    estimator = load_depth_estimator()
    features, t = compose_features(path, estimator, shape=(256, 256), weights=[0.8, 1.2, 0.8, 0.1])
    
    params = {'supper_pixel':2, 'embedding_method':'spectral', 'n_components':10, 'n_neighbors':20, 'cluster_method':'knn', 'n_clusters':7, 'pos_weight':0.5}
    labels_shaped, embedding, labels = segment(features, params=params)
    
    fig, axes = plt.subplots(1, len(t)+1, figsize=(16, 4))

    for i in range(len(t)) : 
        
        axes[i].imshow(t[i])
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i}')
        
    axes[-1].imshow(labels_shaped)
    axes[-1].axis('off')
    axes[-1].set_title(f'Segementation')
    
    plt.savefig('../images/segmentation.png')
    # plt.show()