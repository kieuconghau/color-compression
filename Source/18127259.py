#  Copyright (c) 2020 Kieu Cong Hau

import PIL
from PIL import Image
import numpy as np

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

    # Khởi tạo 2 giá trị trả về của hàm này: labels và centroids.
    labels = [None for _ in range(len(img_1d))]
    centroids = []

    if init_centroids == 'random':
        # Chọn k_clusters màu ngẫu nhiên bất kỳ phân biệt làm trung tâm.
        centroids = [np.random.randint(low=0, high=255, size=3) for _ in range(k_clusters)]
        while len(np.unique(centroids, axis=0)) != k_clusters:
            centroids = [np.random.randint(low=0, high=255, size=3) for _ in range(k_clusters)]
    else:
        # Cập nhật lại k_clusters nếu số lượng màu phân biệt có trong ảnh nhỏ hơn k_clusters.
        distinct_pixels = np.unique(img_1d, axis=0)
        k_clusters = min(k_clusters, len(distinct_pixels))

        # Chọn k_clusters màu phân biệt từ ảnh làm trung tâm.
        random_index_list = np.random.choice(a=len(distinct_pixels), size=k_clusters, replace=False)
        centroids = [distinct_pixels[i] for i in random_index_list]

    # Tìm bộ centroids mới cho đến khi nhãn của các điểm ảnh không có sự thay đổi nữa hoặc đạt lặp mức giới hạn max_iter.
    while max_iter:
        # Copy labels cũ sang biến pre_labels.
        pre_labels = labels.copy()

        # Clusters: Dùng để lưu danh sách index của từng cụm.
        clusters = [[] for _ in range(k_clusters)]

        # Duyệt từng điểm ảnh và gán nhãn cho điểm ảnh đó ứng với trung tâm mà nó gần nhất.
        for i in range(len(img_1d)):
            labels[i] = np.linalg.norm(centroids - img_1d[i], axis=1).argmin()
            clusters[labels[i]].append(i)

        # Nếu không có sự thay đổi nhãn giữa các điểm ảnh, dừng vòng lặp.
        if pre_labels == labels:
            break

        # Duyệt từng cụm, tính lại giá trị trung tâm của cụm đó bằng means (lấy trung bình giữa các điểm ảnh).
        # Nếu cụm bất kỳ không có điểm ảnh nào thuộc về thì giữ lại màu của trung tâm cũ.
        for i in range(k_clusters):
            if clusters[i]:
                centroids[i] = np.mean([img_1d[j] for j in clusters[i]], axis=0)

        # Giảm giới hạn lặp xuống 1 đơn vị cho đến khi max_iter = 0 thì thoát vòng lặp.
        max_iter -= 1

    # Trả về danh sách các màu trung tâm (centroids) và danh sách nhãn của từng điểm ảnh (labels).
    return np.array(centroids).astype(int), np.array(labels)


def color_compression(img_path: str, k_clusters: int, max_iter: int = 1000, init_centroids: str = 'random'):
    """
    Color compression

    Inputs:
        img_path: str
                A path of an image that you want to compress
        k_clusters: int
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

    # Kiểm tra đầu vào hợp lệ.
    if k_clusters < 0:
        print("Oops... k_clusters must be a positive integer!")
        return None

    if max_iter < 0:
        print("Oops... max_iter must be a positive integer!")
        return None

    if init_centroids not in ['random', 'in_pixels']:
        print("Oops... init_centroids must be \'random\' or \'in_pixels\' only")
        return None

    # Đọc nội dung của ảnh từ đường dẫn img_path.
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print("Oops... \"" + img_path + "\" does not exist.")
        return None
    except PIL.UnidentifiedImageError:
        print("Oops... \"" + img_path + "\" is not an image.")
        return None
    except:
        print("Unidentified error!")
        return None

    # Đổi nội dung của ảnh vừa đọc sang dạng RGB, sau đó đổi sang numpy.array và lưu vào img.
    img = img.convert('RGB')
    img = np.array(img)

    # Đổi shape của img thành (w*h, 3).
    img_height, img_width, num_channels = img.shape
    img = img.reshape(img_height * img_width, num_channels)

    # Gom nhóm các điểm ảnh.
    centroids, labels = kmeans(img, k_clusters, max_iter, init_centroids)

    # Cập nhật lại từng điểm ảnh tương ứng với nhóm mà nó thuộc về.
    for i in range(len(img)):
        img[i] = centroids[labels[i]]

    # Trả về nội dung bức ảnh sau khi gom nhóm với shape=(w, h, 3)
    return img.reshape(img_height, img_width, num_channels)


if __name__ == "__main__":
    # Khởi tạo các thông số đầu vào cho hàm nén color_compression.
    img_path = input("Enter an image's path: ")
    if img_path[0] == '\"' and img_path[-1] == '\"':
        img_path = img_path[1:-1]

    k_clusters_list = [3, 5, 7]
    max_iter = 1000
    init_centroids_list = ['random', 'in_pixels']

    # Nén tất cả các hình (6 hình).
    img_list = [[color_compression(img_path, k_clusters, max_iter, init_centroids) for k_clusters in k_clusters_list]
                for init_centroids in init_centroids_list]

    # Lưu lại toàn bộ các hình đã được nén vào thư mục cùng cấp với mã nguồn này.
    if None not in img_list:
        for i in range(len(init_centroids_list)):
            for j in range(len(k_clusters_list)):
                if img_list[i][j] is not None:
                    PIL.Image.fromarray(img_list[i][j], 'RGB').save(init_centroids_list[i] + "_" + str(k_clusters_list[j]) + ".jpg")
