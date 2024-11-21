from functions import hamming_distance
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

text_low = [
    '0'*100+'1'*0,
    '0'*95+'1'*5,
    '0'*90+'1'*10,
    '0'*85+'1'*15,
]
text_mid = [
    "0011001011010011111101000010111111010011000000000001110000001111111110000101110001100000100100010110",
    "1100001010110101111001100010010011100110011000001001110010101011111100010000111000001100100110010011",
    "0110000010110000001101000110011110110110001001010001110110011111111101100000100011001010100110110000",
    "0110000000110101011001010110011111010101111010100000110110011111110100110000110001010010100100010000",
]

text_high = [
    "0011000101100010001100011111001010011010011001111010111100001001011011100010011000111011101101000001",
    "1101111110011110000010100100100100101011101110000111010110100101110111111111011010001000110001111000",
    "0100010000100101011101110111011101110011100111011101000011110011001100010111100100110100011100000100",
    "0000010000100001111000011100100011000110110000101110100000010111010110011011000000110011001110101010"
]


def cluster(flattened_data,n_cluster,method):
    '''
    flatten_data: data with shape of (numsamples,h*w*c) and numpy array type
    method: clustering methods
    n_cluster: the num of clustering centers
    return a label numpy array with shape of (numsamples,)
    '''
    if method == "km":
        kmeans = KMeans(n_clusters=n_cluster,random_state=0,n_init="auto")
        kmeans.fit(flattened_data)

        return kmeans.labels_ , kmeans.cluster_centers_
    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_cluster)
        gmm.fit(flattened_data)
        cluster_labels = gmm.predict(flattened_data)
        return cluster_labels
    elif method == "spectral":
        spectral_clustering = SpectralClustering(n_clusters=n_cluster, affinity='nearest_neighbors', random_state=0)
        cluster_labels = spectral_clustering.fit_predict(flattened_data)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    bit_strings = text_high
    for i in range(len(bit_strings)):
        for j in range(i + 1, len(bit_strings)):
            dist = hamming_distance(bit_strings[i], bit_strings[j])
            print(dist)