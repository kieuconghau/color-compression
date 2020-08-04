#  Copyright (c) 2020 Kieu Cong Hau

import PIL
import numpy as np
import matplotlib.pyplot as plt
from time import time

def kmeans(img_1d: np.ndarray, k_clusters: int, max_iter: int, init_centroids: str):
    """
    K-Means algorithm

    Inputs:
        img_1d : np.ndarray with shape=(height * width, num_channels)
            Original image in 1d array

        k_clusters : int
            Number of clusters

        max_iter : int
            Max iterator

        init_centroids : str
            The way which use to init centroids:
                'random' --> centroid has `c` channels, with `c` is initial random in [0,255]

                'in_pixels' --> centroid is a random pixels of original image

    Outputs:
        centroids : np.ndarray with shape=(k_clusters, num_channels)
            Store color centroids

        labels : np.ndarray with shape=(height * width, )
            Store label for pixels (cluster's index on which the pixel belongs)
    """

    centroids = None
    labels = [None for _ in range(len(img_1d))]

    if init_centroids == 'random':
        centroids = [np.random.randint(low=0, high=255, size=3) for _ in range(k_clusters)]
    else:
        random_index_list = np.random.choice(a=len(img_1d), size=k_clusters, replace=False)
        centroids = [img_1d[i] for i in random_index_list]

    while max_iter:
        start = time()

        pre_labels = labels.copy()

        clusters = [[] for _ in range(k_clusters)]

        for i in range(len(img_1d)):
            centroid_index = np.linalg.norm(centroids - img_1d[i], axis=1).argmin()
            labels[i] = centroid_index
            clusters[centroid_index].append(i)

        if pre_labels == labels:
            break

        for i in range(k_clusters):
            centroids[i] = np.mean([img_1d[j] for j in clusters[i]], axis=0)

        max_iter -= 1

        end = time()
        print(end - start)

    return centroids, labels


def color_compression(img_path: str, num_colors: int, max_iter: int = 1000, init_centroids: str = 'random'):
    """
    Color compression

    Inputs:
        img_path: str
                A path of an image that you want to compress

        num_colors: int
                Number of colors

        max_iter : int
            Max iterator

        init_centroids : str
            The way which use to init centroids:
                'random' --> centroid has `c` channels, with `c` is initial random in [0,255]

                'in_pixels' --> centroid is a random pixels of original image

    Outputs:
        img: np.ndarray with shape=(height, width, num_channels)
            An image in 2d array
    """

    if num_colors < 0:
        print("Oops... num_colors must be a positive integer!")
        return None

    if max_iter < 0:
        print("Oops... max_iter must be a positive integer!")
        return None

    if init_centroids not in ['random', 'in_pixels']:
        print("Oops... init_centroids must be \'random\' or \'in_pixels\' only")
        return None

    try:
        img = PIL.Image.open(img_path)
    except FileNotFoundError:
        print("Oops... \"" + img_path + "\" does not exist.")
        return None
    except PIL.UnidentifiedImageError:
        print("Oops... \"" + img_path + "\" is not an image.")
        return None
    except:
        print("Unidentified error!")
        return None

    img = img.convert('RGB')
    img = np.array(img)

    img_height, img_width, num_channels = img.shape
    img = img.reshape(img_height * img_width, num_channels)

    centroids, labels = kmeans(img, num_colors, max_iter, init_centroids)

    for i in range(len(img)):
        img[i] = centroids[labels[i]]

    img = img.reshape(img_height, img_width, num_channels)

    return img


if __name__ == "__main__":
    start = time()

    path = r"C:\Users\PC\Desktop\06.jpg"
    img_2d = color_compression(path, 7, 1000, 'in_pixels')

    end = time()
    print(end - start)

    if img_2d is not None:
        image = PIL.Image.fromarray(img_2d, 'RGB')
        plt.imshow(image)
        plt.show()
