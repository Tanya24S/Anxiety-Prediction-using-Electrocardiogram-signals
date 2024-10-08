import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/content/All_hrv_features.csv')
X = data.drop(columns=['File_Name', 'Label'])
y = data['Label']
X.dropna(inplace=True)
y = y[X.index]

import numpy as np
import random

class GreyWolfOptimizer:
    def __init__(self, num_agents, max_iter, num_features):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.num_features = num_features
        self.alpha_pos = np.zeros(num_features)
        self.beta_pos = np.zeros(num_features)
        self.delta_pos = np.zeros(num_features)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.random.randint(2, size=(num_agents, num_features))

    def fitness(self, position, X_train, y_train, X_val, y_val):
        selected_features = np.where(position == 1)[0]
        if len(selected_features) == 0:
            return float('inf')
        X_train_fs = X_train[:, selected_features]
        X_val_fs = X_val[:, selected_features]
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train_fs, y_train)
        y_pred = model.predict(X_val_fs)
        return 1.0 - accuracy_score(y_val, y_pred)

    def update_position(self, X_train, y_train, X_val, y_val):
        for i in range(self.num_agents):
            fitness = self.fitness(self.positions[i], X_train, y_train, X_val, y_val)
            if fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness
                self.alpha_pos = self.positions[i].copy()
            elif fitness < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness
                self.beta_pos = self.positions[i].copy()
            elif fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos = self.positions[i].copy()

    def optimize(self, X_train, y_train, X_val, y_val):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(self.num_agents):
                for j in range(self.num_features):
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (X1 + X2 + X3) / 3
                    if random.random() > 0.5:
                        self.positions[i, j] = 1
                    else:
                        self.positions[i, j] = 0

                self.update_position(X_train, y_train, X_val, y_val)

        return self.alpha_pos

"""### **KNeighborsClassifier**"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.shape[1]
gwo = GreyWolfOptimizer(num_agents=10, max_iter=20, num_features=num_features)
best_features = gwo.optimize(X_train.values, y_train.values, X_val.values, y_val.values)

selected_features = np.where(best_features == 1)[0]
X_train_fs = X_train.iloc[:, selected_features]
X_val_fs = X_val.iloc[:, selected_features]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_fs, y_train)
y_pred = model.predict(X_val_fs)

accuracy = accuracy_score(y_val, y_pred)
print(f'GWO Selected Features: {selected_features}')
print(f'Accuracy with GWO Selected Features: {accuracy}')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.shape[1]
gwo = GreyWolfOptimizer(num_agents=10, max_iter=20, num_features=num_features)
best_features = gwo.optimize(X_train.values, y_train.values, X_val.values, y_val.values)

selected_features_indices = np.where(best_features == 1)[0]
selected_features = X.columns[selected_features_indices]
X_train_fs = X_train.iloc[:, selected_features_indices]
X_val_fs = X_val.iloc[:, selected_features_indices]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_fs, y_train)
y_pred = model.predict(X_val_fs)

accuracy = accuracy_score(y_val, y_pred)
print(f'GWO Selected Features: {selected_features}')
print(f'Accuracy with GWO Selected Features: {accuracy}')
selected_data = data[['File_Name', 'Label'] + list(selected_features)]
selected_data.to_csv('selected_features.csv', index=False)

from sklearn.metrics import classification_report, confusion_matrix

classification=classification_report(y_val, y_pred)
confusion=confusion_matrix(y_val, y_pred)
print(f'Classification Report:\n {classification}')
print(f'Confusion Matrix:\n {confusion}')

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

"""### **Trying to visualize**"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/content/All_hrv_features.csv')
X = data.drop(columns=['File_Name', 'Label'])
y = data['Label']
X.dropna(inplace=True)
y = y[X.index]

import numpy as np
import random

class GreyWolfOptimizer:
    def __init__(self, num_agents, max_iter, num_features):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.num_features = num_features
        self.alpha_pos = np.zeros(num_features)
        self.beta_pos = np.zeros(num_features)
        self.delta_pos = np.zeros(num_features)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.random.randint(2, size=(num_agents, num_features))

    def fitness(self, position, X_train, y_train, X_val, y_val):
        selected_features = np.where(position == 1)[0]
        if len(selected_features) == 0:
            return float('inf')
        X_train_fs = X_train[:, selected_features]
        X_val_fs = X_val[:, selected_features]
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train_fs, y_train)
        y_pred = model.predict(X_val_fs)
        return 1.0 - accuracy_score(y_val, y_pred)

    def update_position(self, X_train, y_train, X_val, y_val):
        for i in range(self.num_agents):
            fitness = self.fitness(self.positions[i], X_train, y_train, X_val, y_val)
            if fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness
                self.alpha_pos = self.positions[i].copy()
            elif fitness < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness
                self.beta_pos = self.positions[i].copy()
            elif fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos = self.positions[i].copy()

    def optimize(self, X_train, y_train, X_val, y_val):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(self.num_agents):
                for j in range(self.num_features):
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (X1 + X2 + X3) / 3
                    if random.random() > 0.5:
                        self.positions[i, j] = 1
                    else:
                        self.positions[i, j] = 0

                self.update_position(X_train, y_train, X_val, y_val)

        return self.alpha_pos

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.shape[1]
gwo = GreyWolfOptimizer(num_agents=40, max_iter=80, num_features=num_features)
best_features = gwo.optimize(X_train.values, y_train.values, X_val.values, y_val.values)

selected_features_indices = np.where(best_features == 1)[0]
selected_features = X.columns[selected_features_indices]
X_train_fs = X_train.iloc[:, selected_features_indices]
X_val_fs = X_val.iloc[:, selected_features_indices]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_fs, y_train)
y_pred = model.predict(X_val_fs)

accuracy = accuracy_score(y_val, y_pred)
print(f'GWO Selected Features: {selected_features}')
print(f'Accuracy with GWO Selected Features: {accuracy}')
selected_data = data[['File_Name', 'Label'] + list(selected_features)]
selected_data.to_csv('selected_features.csv', index=False)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data_file=pd.read_csv('/content/selected_features.csv')
data=data_file.iloc[:,2:]
correlation_matrix = data.corr()
print(correlation_matrix)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of HRV Features')
plt.show()

"""### **Convergence curve**"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt

class GreyWolfOptimizer:
    def __init__(self, num_agents, max_iter, num_features):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.num_features = num_features
        self.alpha_pos = np.zeros(num_features)
        self.beta_pos = np.zeros(num_features)
        self.delta_pos = np.zeros(num_features)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.random.randint(2, size=(num_agents, num_features))
        self.convergence_curve = []

    def fitness(self, position, X_train, y_train, X_val, y_val):
        selected_features = np.where(position == 1)[0]
        if len(selected_features) == 0:
            return float('inf')
        X_train_fs = X_train[:, selected_features]
        X_val_fs = X_val[:, selected_features]
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train_fs, y_train)
        y_pred = model.predict(X_val_fs)
        return 1.0 - accuracy_score(y_val, y_pred)

    def update_position(self, X_train, y_train, X_val, y_val):
        for i in range(self.num_agents):
            fitness = self.fitness(self.positions[i], X_train, y_train, X_val, y_val)
            if fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness
                self.alpha_pos = self.positions[i].copy()
            elif fitness < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness
                self.beta_pos = self.positions[i].copy()
            elif fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos = self.positions[i].copy()

    def optimize(self, X_train, y_train, X_val, y_val):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(self.num_agents):
                for j in range(self.num_features):
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (X1 + X2 + X3) / 3
                    if random.random() > 0.5:
                        self.positions[i, j] = 1
                    else:
                        self.positions[i, j] = 0

                self.update_position(X_train, y_train, X_val, y_val)

            self.convergence_curve.append(self.alpha_score)

        return self.alpha_pos

data = pd.read_csv('/content/All_hrv_features.csv')
X = data.drop(columns=['File_Name', 'Label'])
y = data['Label']
X.dropna(inplace=True)
y = y[X.index]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.shape[1]
gwo = GreyWolfOptimizer(num_agents=10, max_iter=20, num_features=num_features)
best_features = gwo.optimize(X_train.values, y_train.values, X_val.values, y_val.values)

selected_features = np.where(best_features == 1)[0]
X_train_fs = X_train.iloc[:, selected_features]
X_val_fs = X_val.iloc[:, selected_features]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_fs, y_train)
y_pred = model.predict(X_val_fs)

accuracy = accuracy_score(y_val, y_pred)
print(f'GWO Selected Features: {selected_features}')
print(f'Accuracy with GWO Selected Features: {accuracy}')

#convergence curve
plt.plot(gwo.convergence_curve)
plt.title('Convergence Curve of GWO')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.show()
