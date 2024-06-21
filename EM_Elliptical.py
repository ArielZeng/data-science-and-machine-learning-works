import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.stats import multivariate_normal
import rpy2.robjects as robjects
from rpy2.rinterface_lib.callbacks import consolewrite_warnerror
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import warnings
import os
import matplotlib.pyplot as plt

# Suppress R warnings
def custom_rpy2_warn_handler(message, category, filename, lineno, file=None, line=None):
    pass


warnings.showwarning = custom_rpy2_warn_handler
os.environ["LANG"] = "en_US.UTF-8"
numpy2ri.activate()


def mvstable_define_calling_R(alpha, delta, R):
    robjects.globalenv["alpha"] = alpha
    robjects.globalenv["delta"] = delta
    robjects.globalenv["R"] = R
    r_code = '''
    library(stable)
    ret <- mvstable.elliptical(alpha = alpha, delta = delta, R = R)
    '''
    robjects.r(r_code)
    # 操作成功完成，返回True
    return True

def mvstable_fit_calling_R(cov_matrix, method_num):

    robjects.globalenv["x"] = cov_matrix
    r_code = '''
    library(stable)
    ret<-mvstable.fit.elliptical(x, method1d=1)
    '''
    robjects.r(r_code)
    # 检查是否有 'result' 对象
    alpha = robjects.r('ret["alpha"]')
    delta = robjects.r('ret["delta"]')
    R = robjects.r('ret["R"]')
    R = nearpd(R[0], corr=True)
    eigenvalues = robjects.r('ret["eigenvalues"]')
    normF = robjects.r('ret["normF"]')
    method1d = robjects.r('ret["method1d"]')
    #print("alpha: ", alpha[0])
    #print("delta: ", delta[0])
    return alpha[0], delta[0], R, eigenvalues[0], normF[0], method1d[0]


def nearpd(x, corr=True):
    robjects.globalenv["cov_matrix"] = x
    r_code = '''
    library("Matrix")
    ret<-nearPD(cov_matrix, corr = TRUE) 
    '''
    robjects.r(r_code)
    mat = robjects.r('ret["mat"]')
    return mat[0]

def dmvstable_elliptical_calling_R(val, alpha, delta, R):
    #print("dmvstable_elliptical_calling_R called")
    if val.ndim == 1:
        #print(f"Original shape: {val.shape}")
        val = val[:, np.newaxis]
        #print(f"New shape: {val.shape}")
    robjects.globalenv["alpha"] = alpha
    robjects.globalenv["delta"] = delta
    robjects.globalenv["R"] = R
    robjects.globalenv["data_matrix"] = np.array(val)
    r_code = '''
    library(stable)
    probabilities <- apply(data_matrix, 1, function(x) {
         # 将行向量 x 转换为 1 x ncol(data_matrix) 的矩阵
         x_matrix <- matrix(x, nrow = 1, ncol = ncol(data_matrix))
         # Alternatively, use transpose
         x_matrix_transposed <- t(matrix(x, nrow = 1, ncol = ncol(data_matrix)))
         # 调用 dmvstable.elliptical 函数
         dmvstable.elliptical(x = x_matrix_transposed, alpha = alpha, R = R, delta = delta)
     })

    '''
    robjects.r(r_code)
    probabilities = robjects.r('probabilities')
    probabilities = np.array(probabilities)

    return probabilities

# centroids = np.array([
#     [-1.02184904, -0.1249576,  -1.227541,   -1.31297673 ],
#     [-0.41600969, -1.28197243,  0.1372359,   0.13322594],
#     [0.79566902, -0.1249576,   0.81962435,  1.05353673],
#     [-0.90068117,  1.72626612, -1.227541,   -1.31297673],
#     [0.31099753, -0.35636057,  0.53529583,  0.26469891]
# ])


def assign_to_nearest_centroid(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def calculate_initial_variances(data, labels, centroids):
    k = len(centroids)
    variances = []
    for i in range(k):
        cluster_data = data[labels == i]
        if cluster_data.shape[0] > 1:
            variance = np.var(cluster_data, axis=0, ddof=1)
        else:
            # Fallback to identity matrix if the cluster has fewer than two points
            variance = np.ones(data.shape[1])
        variances.append(variance)
    return variances

def calculate_initial_covariances(data, labels, centroids):
    k = len(centroids)
    covariances = []
    for i in range(k):
        cluster_data = data[labels == i]
        if cluster_data.shape[0] > 1:
            covariance = np.cov(cluster_data, rowvar=False)
        else:
            # Fallback to identity matrix if the cluster has fewer than two points
            covariance = np.eye(data.shape[1])
        covariances.append(covariance)
    return covariances

class Elliptical_Distribution:
    def __init__(self, alpha, delta, R):
        self.alpha = alpha
        self.delta = delta
        self.R = R

    def fit(self, data):
        # 确保data是二维数组
        if len(data.shape) == 1:
            data = data[np.newaxis, :]

        # 添加正则化项以避免数值稳定性问题
        reg_term = np.eye(data.shape[1]) * 1e-6
        # 计算协方差矩阵并添加正则化项
        cov_matrix = np.cov(data, rowvar=False) + reg_term

        # 使用mvstable.fit.elliptical方法更新alpha, delta和R
        fit_result = mvstable_fit_calling_R(cov_matrix, 1)
        self.alpha, self.delta, self.R = fit_result[:3]

        # 打印更新后的参数值
        print("Updated alpha: ", self.alpha)
        print("Updated delta: ", self.delta)
        print("Updated R: \n", self.R)

    def pdf(self, data):
        # 确保数据是二维数组形式
        # 如果data是一维数组（单个数据点的情况），将其转换为二维数组
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # 确保alpha值在有效范围内
        alpha_min = 1  # 定义alpha的最小有效值
        alpha_max = 2  # 定义alpha的最大有效值
        alpha_adjusted = max(alpha_min, min(self.alpha, alpha_max))  # 调整alpha值
        probabilities = dmvstable_elliptical_calling_R(data, alpha_adjusted, self.delta, self.R)

        return probabilities[0] if probabilities.size == 1 else probabilities

    def __repr__(self):
        return 'Elliptical_Distribution({}, {}, {})'.format(self.alpha, self.delta, self.R)

class EllipticalMixture_self:
    def __init__(self, data, k, centroids=None, covariances=None, mix_init=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        self.data = data
        self.n_components = k

        # Initialize centroids if None are provided
        if centroids is None:
            # Randomly initialize the centroids within the range of the dataset
            min_max = np.array([data.min(axis=0), data.max(axis=0)])
            self.centroids = np.random.uniform(min_max[0], min_max[1], (k, data.shape[1]))
        else:
            self.centroids = centroids

        # Initialize covariance matrices if None are provided
        if covariances is None:
            self.covariances = [np.eye(data.shape[1]) for _ in range(k)]
        else:
            self.covariances = covariances

        # 使用K-Means质心初始化均值，如果提供了方差和协方差则使用它们，否则使用单位矩阵
        self.distributions = [
            Elliptical_Distribution(
                alpha = np.random.uniform(1, 2),  # 假定初始alpha范围
                delta = self.centroids[i],
                R = self.covariances[i]
            ) for i in range(k)
        ]

        # 如果提供了混合系数则使用它们，否则均匀初始化
        self.mix = mix_init if mix_init is not None else np.ones(k) / k
        self.loglike = -np.inf

    def get_centroids(self):
        """
        提取模型当前迭代中使用的质心。

        返回:
        - centroids: 一个包含所有质心的列表或数组。
        """
        # 假设每个分布的delta属性表示其中心
        centroids = [distribution.delta for distribution in self.distributions]
        return np.array(centroids)

    def Estep(self):
        #print("Entered Estep")
        N = len(self.data)
        K = self.n_components  # 分布数量
        weights = np.zeros((N, K))
        posterior_probs = np.zeros((N, K))

        # Iterate over each distribution component
        for k, distribution in enumerate(self.distributions):
            # Compute the probability density for each data point under current distribution
            for i, datum in enumerate(self.data):
                posterior_probs[i, k] = distribution.pdf(datum) * self.mix[k]

        # Sum of posterior probabilities for normalization
        total_probs = np.sum(posterior_probs, axis=1)

        # Avoid division by zero
        total_probs[total_probs == 0] = 1e-9
        self.loglike = np.sum(np.log(total_probs))
        # Normalize the posterior probabilities to get the weights
        weights = posterior_probs / total_probs[:, None]

        # Update labels to the index of the maximum weight
        self.labels = np.argmax(weights, axis=1)

        # Return the weights for use in the M step
        return weights

    def Mstep(self, weights):
        #print("Entered Mstep")
        N, D = self.data.shape  # 数据点数量和特征维度
        K = self.n_components  # 分布数量

        for k in range(K):
            weight_k = weights[:, k]  # 获取第k个分布的权重
            total_weight = weight_k.sum()
            self.mix[k] = total_weight / N
            self.distributions[k].mu = np.sum(self.data * weight_k[:, np.newaxis], axis=0) / total_weight
            diff = self.data - self.distributions[k].mu
            self.distributions[k].weighted_cov = np.dot(weight_k * diff.T, diff) / total_weight + np.eye(D) * 1e-6

            # 使用加权协方差矩阵重新估计分布参数
            fit_result = mvstable_fit_calling_R(self.distributions[k].weighted_cov, 1)

            # 更新分布参数
            self.distributions[k].alpha = fit_result[0]
            self.distributions[k].delta = fit_result[1]
            self.distributions[k].R = fit_result[2]

    def iterate(self, N=1, verbose=True, tolerance=10e-0):
        previous_loglike = -np.inf  # 初始化为负无穷大
        for i in range(1, N + 1):
            self.Mstep(self.Estep())  # 执行一次E步骤和M步骤
            current_wcss = self.calculate_wcss()
            # 日志似然值已在E步骤或M步骤中更新至self.loglike
            if verbose:
                print(f'Iteration {i}: WCSS = {current_wcss}, log-likelihood = {self.loglike}')
            # 绘制当前迭代的聚类结果
            # visualize_clusters(self.data, self, f"Iteration {i} clusters")

            # 检查收敛
            if i > 1 and abs(self.loglike - previous_loglike) <= tolerance:
                if verbose:
                    print(f'Converged at iteration {i}.')
                break  # 如果满足收敛条件，则退出循环

            previous_loglike = self.loglike  # 更新前一个日志似然值为当前值

        # 迭代完成后，打印最终的分布参数
        if verbose:
            for idx, distribution in enumerate(self.distributions):
                print(
                    f'Distribution {idx}: alpha = {distribution.alpha}, delta = {distribution.delta}, R = {distribution.R}')

        # 最后执行一次E步骤以更新最终的权重和标签
        self.Estep()

    def calculate_wcss(self):
        """
        Calculate the Within-Cluster Sum of Squares (WCSS).

        Parameters:
        - data: 数据点的集合。
        - centroids: 聚类中心的集合。
        - labels: 数据点的聚类分配标签。

        Returns:
        - wcss: 簇内平方和。
        """
        centroids = self.get_centroids()  # 确保有这个方法来获取当前的质心
        wcss = 0
        for i, centroid in enumerate(centroids):
            cluster_points = self.data[self.labels == i]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            wcss += np.sum(distances ** 2)
        return wcss

    def pdf(self, x):
        # 计算并返回给定数据点在混合模型下的总概率密度
        total_pdf = sum(
            mix_weight * distribution.pdf(x) for mix_weight, distribution in zip(self.mix, self.distributions))
        return total_pdf

    def predict(self, datum):
        """
        Predict the cluster for a given data point based on highest probability density.
        """
        # 计算每个分布对于给定数据点的概率密度，考虑混合权重
        probs = [distribution.pdf(datum) * mix_weight for mix_weight, distribution in zip(self.mix, self.distributions)]
        # 选择并返回具有最高概率密度的分布的索引
        return np.argmax(probs)

    def predict_clusters(self, data):
        """
        Assign clusters to each data point in the dataset.
        """
        return np.array([self.predict(datum) for datum in data])

    def __repr__(self):
        distributions_repr = ', '.join(repr(dist) for dist in self.distributions)
        return f'EllipticalMixture({distributions_repr}, mix={self.mix})'

    def __str__(self):
        distributions_str = ', '.join(str(dist) for dist in self.distributions)
        return f'Mixture: {distributions_str}, mix={self.mix}'

colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'gray']
def visualize_clusters(data, labels, centroids=None, title=''):
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Visualization
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)

    # Colors mapped to initial true labels or predicted clusters
    if centroids is not None:
        # If centroids are provided, then use predicted cluster labels
        reduced_centroids = pca.transform(centroids)
        for i, centroid in enumerate(reduced_centroids):
            cluster_data = reduced_data[labels == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], edgecolor=None, s=40,
                        label=f'Cluster {i + 1}')
            plt.scatter(centroid[0], centroid[1], c='black', marker='x', s=150, label=f'Centroid {i + 1}',
                        edgecolor='k')
    else:
        # Use true labels for colors
        for i, label in enumerate(unique_labels):
            plt.scatter(reduced_data[labels == label, 0], reduced_data[labels == label, 1], c=colors[i], edgecolor=None,
                        s=40, label=f'Class {i + 1}')

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()


def find_best_Elliptical(data, n_clusters=None, try_how_many_times=1, mix_init=None, random_state=None):
    best_model = None
    best_loglike = float('-inf')
    best_iteration = -1

    # Try multiple times to find the best Elliptical Mixture Model based on log-likelihood
    for try_index in range(try_how_many_times):
        # Initialize the Elliptical Mixture Model with K-Means++ results
        mix = EllipticalMixture_self(data, k=n_clusters, mix_init=mix_init, random_state=random_state)

        # Train the model with a convergence criterion within mix.iterate
        try:
            # Iterate with a high upper bound and check for convergence within the iterate method
            mix.iterate(N=1000, verbose=True, tolerance=10e-0)
            current_loglike = mix.loglike
            if current_loglike > best_loglike:
                best_loglike = current_loglike
                best_mix = mix
                best_iteration = try_index
        except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
            print(f"One trial less due to errors: {e}")
            continue

    if best_mix is not None:
        print(f'Best model found in iteration {best_iteration + 1} with log-likelihood {best_loglike}')
    else:
        print("No successful GMM optimization was found.")

    return best_mix

def calculate_confusion_matrix(y_true, y_pred, n_clusters):
    """
    Create a confusion matrix and calculate the purity of each predicted cluster.

    Parameters:
    - y_true: array-like, true class labels
    - y_pred: array-like, predicted cluster labels
    - n_clusters: int, number of clusters

    Returns:
    - confusion_matrix: 2D array, the confusion matrix
    """
    # Initialize confusion matrix
    confusion_matrix = np.zeros((n_clusters, n_clusters))

    # Populate the confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label][pred_label] += 1

    return confusion_matrix

# Function to calculate individual cluster purity
def calculate_weighted_cluster_purity(confusion_mat, n_clusters):
    """
    Calculate the weighted purity of the entire clustering based on the confusion matrix.

    Parameters:
    - confusion_mat: 2D array, the confusion matrix
    - n_clusters: int, number of clusters

    Returns:
    - weighted_purity: float, weighted average purity of the clustering
    """
    # Calculate the total number of samples
    total_samples = np.sum(confusion_mat)

    # Initialize purity to zero
    weighted_purity = 0

    for i in range(n_clusters):
        # Maximum count of any label in the cluster
        max_label_count = np.max(confusion_mat[:, i])

        # Total count of all samples in the cluster
        cluster_size = np.sum(confusion_mat[:, i])

        # Purity for this cluster
        if cluster_size > 0:
            cluster_purity = max_label_count / cluster_size
        else:
            cluster_purity = 0

        # Weight this cluster's purity by its size relative to the total
        weighted_purity += (cluster_size / total_samples) * cluster_purity

    return weighted_purity

from sklearn.metrics import silhouette_samples, silhouette_score
def visualize_silhouette_scores(n_clusters, X, labels, cluster_labels):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

from sklearn.metrics import pairwise_distances
def dunn_index(X, labels):
    """
    Compute the Dunn Index for a set of clusters.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.

    Returns:
    di : float
        The Dunn Index, higher values indicate better clustering.
    """
    # Unique clusters
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return 0  # Dunn index is not defined for less than two clusters

    # Compute distance matrix
    distances = pairwise_distances(X)

    # Initialize inter-cluster and intra-cluster distances
    min_inter_cluster_distance = np.inf
    max_intra_cluster_distance = 0

    # Compute minimum inter-cluster distance
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster_i = X[labels == clusters[i]]
            cluster_j = X[labels == clusters[j]]
            inter_cluster_distance = np.min(pairwise_distances(cluster_i, cluster_j))
            min_inter_cluster_distance = min(min_inter_cluster_distance, inter_cluster_distance)

    # Compute maximum intra-cluster distance
    for i in clusters:
        cluster_i = X[labels == i]
        intra_cluster_distance = np.max(pairwise_distances(cluster_i))
        max_intra_cluster_distance = max(max_intra_cluster_distance, intra_cluster_distance)

    # Calculate Dunn Index
    di = min_inter_cluster_distance / max_intra_cluster_distance
    return di


from itertools import combinations
def rand_index_score(clusters, classes):
    """
    Calculate the Rand Index (RI) for two clusterings.

    Parameters:
    clusters : array-like, shape (n_samples,)
        Predicted labels for each sample.
    classes : array-like, shape (n_samples,)
        True labels for each sample.

    Returns:
    ri : float
        The Rand Index score.
    """
    tp_plus_fp = sum([1 for i, j in combinations(range(len(clusters)), 2) if clusters[i] == clusters[j]])
    tp_plus_fn = sum([1 for i, j in combinations(range(len(classes)), 2) if classes[i] == classes[j]])
    tp = sum([1 for i, j in combinations(range(len(clusters)), 2) if
              clusters[i] == clusters[j] and classes[i] == classes[j]])
    tn = sum([1 for i, j in combinations(range(len(clusters)), 2) if
              clusters[i] != clusters[j] and classes[i] != classes[j]])

    # Rand Index formula
    ri = (tp + tn) / (tp_plus_fp + tp_plus_fn - tp + tn)
    return ri

# 加载数据集
iris = pd.read_csv("D:/course/P/Project2/6/_Customer Churn/Customer Churn.csv")

# Prepare data
X = iris.iloc[:, :-1]
y = iris['Churn']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# Define the number of clusters
n_clusters = 5
random_state = 0

print("Initial state with true labels:")
visualize_clusters(X_scaled, y, title='Customer Churn: Initial State with True Labels')

best_elliptical = find_best_Elliptical(X_scaled, n_clusters=n_clusters, try_how_many_times=1, mix_init=None, random_state=random_state)

print("true labels:\n", y)
# Predict clusters
predicted_clusters = best_elliptical.predict_clusters(X_scaled)
print("Final state with predicted labels:\n", predicted_clusters)
# Confusion Matrix
confusion_mat = calculate_confusion_matrix(y, predicted_clusters, n_clusters)
print("Confusion Matrix:\n", confusion_mat)

weighted_purity = calculate_weighted_cluster_purity(confusion_mat, n_clusters)
print("Weighted Average Purity of Clustering:", weighted_purity)

# Calculate the silhouette scores and average
silhouette_avg = silhouette_score(X_scaled, predicted_clusters)
print('The average silhouette score is:', silhouette_avg)

# Visualize the silhouette scores
visualize_silhouette_scores(n_clusters, X_scaled, y, predicted_clusters)

dunn_index_value = dunn_index(X_scaled, predicted_clusters)
print("Dunn Index: ", dunn_index_value)

rand_index = rand_index_score(predicted_clusters, y)
print(f"Rand Index: {rand_index}")

# Finally, visualize the predicted clusters after the GMM has been trained
print("Final optimized Elliptical clusters:")
visualize_clusters(X_scaled, predicted_clusters, centroids=best_elliptical.get_centroids(), title='Customer Churn: Final Optimized Elliptical Clusters')
