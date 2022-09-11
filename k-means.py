"""
k-means.py

This Python script implements the K-Means algorithm to test weed segmentation on a 3D point cloud.

Author: Miguel Muñoz

Date of Creation: 07 September 2022
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def main():
    data_folder = "samples/"
    dataset = "small-weed-samples.ply"
    x, y, z, r, g, b = np.loadtxt(data_folder + dataset, skiprows=13, unpack=True)

    r /= 255
    g /= 255
    b /= 255
    RGB = np.dstack((r, g, b))

    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.scatter(x, z, c=RGB.reshape(-1, 3), s=0.05)
    plt.axhline(y=np.mean(z), color='r', linestyle ='-')
    plt.title("First view")
    plt.xlabel('X - axis')
    plt.ylabel('Z - axis')

    plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    plt.scatter(y, z, c=RGB.reshape(-1, 3), s=0.05)
    plt.axhline(y=np.mean(z), color='r', linestyle='-')
    plt.title("Second view")
    plt.xlabel('Y - axis')
    plt.ylabel('Z - axis')

    plt.figure()
    X = np.column_stack((x, y, z, r, g, b))
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    plt.scatter(x, y, c=kmeans.labels_, s=0.1)

    result_folder = "samples/results/"
    np.savetxt(result_folder + dataset.split("/")[0] + "_result_4_clusters.xyz", np.column_stack(
        (x, y, z, kmeans.labels_)), fmt='%1.4f')


if __name__ == '__main__':
    main()
