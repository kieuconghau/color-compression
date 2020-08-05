#  Copyright (c) 2020 Kieu Cong Hau

from ColorCompression import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path = r"C:\Users\PC\Desktop\Australia.jpg"         # Change your image's path here
    k_clusters_list = [3, 5, 7]
    max_iter = 1000
    init_centroids_list = ['random', 'in_pixels']

    img = [[color_compression(img_path, k_clusters, max_iter, init_centroids) for k_clusters in k_clusters_list]
           for init_centroids in init_centroids_list]

    for i in range(len(init_centroids_list)):
        for j in range(len(k_clusters_list)):
            if img[i][j] is not None:
                print(init_centroids_list[i], k_clusters_list[j])
                image = PIL.Image.fromarray(img[i][j], 'RGB')
                plt.imshow(image)
                plt.show()
