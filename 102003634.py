import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as r
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv('dataset.csv')

X = df.drop("Action", axis='columns')
y = df["Action"]

kernalList = ['linear', 'poly', 'rbf', 'sigmoid']
bestAccuracy = 0
cvgs_data = []
all_samples = []

def get_accuracy(X_train, y_train, X_test, y_test, params):
    c = SVC(kernel=params[0], C=params[1], gamma=params[2], degree=params[3])
    c.fit(X_train, y_train)
    y_pred = c.predict(X_test)
    return accuracy_score(y_test, y_pred)


def sample(X, y, iter=1000):
    global bestAccuracy
    sample_bestAccuracy = 0
    sample_bestGamma = 0  
    sample_bestKernel = ''
    sample_bestC = 0
    sample_bestDegree = 0  
    sample_cvgs_data = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    for _ in range(iter):
        kernel = r.choice(kernalList)
        c = r.randint(1, 7)
        g = r.randint(-1, 7)
        p = r.randint(1, 7)
        if (g < 1):
            g = r.choice(['scale', 'auto'])
        if (kernel == 'poly'):
            g = r.choice(['scale', 'auto'])

        accu = get_accuracy(X_train, y_train, X_test,
                            y_test, [kernel, c, g, p])

        if (accu > sample_bestAccuracy):
            sample_bestAccuracy = accu
            sample_bestC = c
            sample_bestDegree = p
            sample_bestKernel = kernel
            sample_bestGamma = g

        sample_cvgs_data.append(sample_bestAccuracy)

    all_samples.append([sample_bestKernel, sample_bestC, sample_bestGamma, sample_bestDegree, sample_bestAccuracy])
    if (sample_bestAccuracy > bestAccuracy):
        global cvgs_data
        cvgs_data = sample_cvgs_data
        bestAccuracy = sample_bestAccuracy


for _ in range(10):
    sample(X, y, 1000)

all_samples = pd.DataFrame(all_samples, columns=[
                           'Kernel', 'c', 'gamma', 'degree', 'Accuracy'])

print("Resultant Table")
print(all_samples)


all_samples.to_csv('./result.csv', index=False)

plt.plot(np.arange(len(cvgs_data)), cvgs_data)
plt.title('Convergence graph of best SVM')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()