from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
import pandas as pd
# Import packages to do the classifying
import numpy as np
from sklearn.svm import SVC

def disease(homoginity, entropy):
    measure = np.array([homoginity,entropy])
    measure = measure.reshape(1,-1)
    
    if(clf.predict(measure)==0):
        print('Bacteria')
        
    elif(clf.predict(measure)==1):
             print('Virus')
        
    else:
           print('Fungus')


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
df= pd.read_csv("C:\\Users\\Ashu\\Desktop\\Books\\Semester project\\min\\data.csv")
df.columns = df.columns.to_series().apply(lambda x: x.strip())
    

x = np.array(df.drop(['Type'],1))
y = np.array(df['Type'])

clf= SVC(kernel='rbf', random_state=0, gamma=0.01, C=100000)
# Train the classifier
clf.fit(x, y)   
        
# Visualize the decision boundaries
plot_decision_regions(x, y, classifier=clf)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

disease(5,28)

        
        