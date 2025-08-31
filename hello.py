import numpy as np
from sklearn.decomposition import PCA

# 生成示例数据
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# 创建PCA对象，降到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("主成分分析后的数据：")
print(X_pca)