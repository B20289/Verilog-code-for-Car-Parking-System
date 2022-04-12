import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

file = pd.read_csv("D:\ACADEMIC STUFF\IC272-DATA SCIENCE 3\Iris.csv")
listall = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']  
x = file.loc[:, listall].values

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

eigenvalue_x,eigenvector_x= np.linalg.eig(np.cov(x.T))
l = np.linspace(1,4,4)

plt.plot(l,[round(i,3) for i in eigenvalue_x])
plt.xticks(np.arange(min(l), max(l)+1, 1.0))
plt.xlabel('components')
plt.ylabel('eigenval')
plt.title('Eigenvalue vs. components')
plt.show()
# 2
K = 3
kmeans = KMeans(n_clusters=K)
kmeans.fit(principalDf)
labels = kmeans.predict(principalDf)
kmeans_prediction = kmeans.predict(principalDf)
principalDf['k_cluster'] = kmeans.labels_

#Getting the Centroids
centroids = kmeans.cluster_centers_

#Getting unique labels
u_labels = np.array(['Iris-versicolor','Iris-setosa','Iris-virginica'])

plt.scatter(principalDf[principalDf.columns[0]], principalDf[principalDf.columns[1]], c=labels, cmap='rainbow', s=15)
plt.scatter([centroids[i][0] for i in range(K)], [centroids[i][1] for i in range(K)], c='black', marker='o',
            label='cluster centres')
plt.legend()
plt.title('Data Points K-Means')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# 2(b)
print('The distortion measure for k =3 is', round(kmeans.inertia_, 3))

# 2(c)
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)  # print(contingency_matrix)
    # print(contingency_matrix)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

orginal = []
for i in file['Species']:
    if i == 'Iris-setosa':
        orginal.append(0)
    elif i == 'Iris-virginica':
        orginal.append(2)
    else:
        orginal.append(1)
print('The purity score for k =3 is', round(purity_score(orginal, labels), 3))
# 3
principalDf = principalDf.drop(['k_cluster'], axis=1)

Ks = [2, 3, 4, 5, 6, 7]
kdistortion = []
kpurity = []
for k in Ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(principalDf)
    kdistortion.append(round(kmeans.inertia_, 3))
    kpurity.append(round(purity_score(orginal, kmeans.predict(principalDf)), 3))

print('The distortion measures are', kdistortion)
print('The purity scores are', kpurity)

plt.plot(Ks, kdistortion)
plt.title('Distortion Measure vs K')
plt.xlabel('K')
plt.ylabel('Distortion Measure')
plt.show()
# 4
# building gmm
gmm = GaussianMixture(n_components=K, random_state=42).fit(principalDf)
gmm_pred = gmm.predict(principalDf)
principalDf['gmm_cluster'] = gmm_pred
gmmcentres = gmm.means_

# plotting the scatter plot
plt.scatter(principalDf[principalDf.columns[0]], principalDf[principalDf.columns[1]], c=gmm_pred, cmap='rainbow', s=15)
plt.scatter([gmmcentres[i][0] for i in range(K)], [gmmcentres[i][1] for i in range(K)], c='black', marker='o', label='cluster centres')
plt.legend()
plt.title('Data Points(GMM)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

principalDf = principalDf.drop(['gmm_cluster'], axis=1)

print('The distortion measure for k =3 is', round(gmm.score(principalDf) * len(principalDf), 3))
print('The purity score for k =3 is', round(purity_score(orginal, gmm_pred), 3))
total_log = []
gmpurity = []
for k in Ks:
    gmm = GaussianMixture(n_components=k, random_state=42).fit(principalDf)
    total_log.append(round(gmm.score(principalDf) * len(principalDf), 3))
    gmpurity.append(round(purity_score(orginal, gmm.predict(principalDf)), 3))

print('The distortion measures are', total_log)
print('The purity scores are', gmpurity)
# plotting K vs distortion measure
plt.plot(Ks, total_log)
plt.title('Distortion Measure vs K')
plt.xlabel('K')
plt.ylabel('Distortion Measure')
plt.show()

eps = [1, 1, 5, 5]
min_samples = [4, 10, 4, 10]
for i in range(4):
    dbscan_model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(principalDf)
    DBSCAN_predictions = dbscan_model.labels_
    print(f'Purity score for eps={eps[i]} and min_samples={min_samples[i]} is', round(purity_score(orginal, DBSCAN_predictions), 3))
plt.scatter(principalDf[principalDf.columns[0]], principalDf[principalDf.columns[1]], c=DBSCAN_predictions, cmap='flag', s=15)
plt.title(f'Data Points for eps={eps[i]} and min_samples={min_samples[i]}')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()