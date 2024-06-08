import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import ShuffleSplit,  cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
# Крок 1
file_name = 'dataset2_l4.txt' 
df = pd.read_csv(file_name) 

# Крок 2
num = df.shape
num_rows, num_columns = num
print(f"Кількість записів: {num_rows}")
print(f"Кількість полів: {num_columns}")

# Крок 3
print(f"Атрибути набору даних: \n{df.columns}")

# Крок 4
counts_class = df.groupby('Class')['Class'].value_counts()
print(counts_class)
# Очевидно, не є збалансованою

# Крок 5
X = df.iloc[:, 0:-1] # вихідні аргументи
y = df.iloc[:, -1] # цільова характеристика
print(f'Вихідні аргументи: \n{X}')
print(f'Цільова характеристика: \n{y}')

rs = ShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
print(rs.get_n_splits(X, y))

variant = int(input("Виберіть номер варіанту (від 1 до 20): ")) - 1
if variant < 0 or variant >= 20:
    print("Невірний номер варіанту. Будь ласка, виберіть номер від 1 до 20.")
    exit()

for i, (train_index, test_index) in enumerate(rs.split(X)):
    if i == variant:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}") 
        break

print(X_train.shape, X_test.shape)


# Крок 6
# print(df.describe())
# df.boxplot()
# plt.show()

k_values = range(2,30)
balanced_accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, scoring='balanced_accuracy')
    balanced_accuracy_scores.append(scores.mean())

optimal_k = k_values[np.argmax(balanced_accuracy_scores)]
print(f"Оптимальне значення k: {optimal_k}")

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Графічне представлення результатів перехресної перевірки
plt.figure(figsize=(10, 6))
plt.plot(k_values, balanced_accuracy_scores, marker='o')
plt.title('Перехресна перевірка для вибору k в KNN (balanced_accuracy)')
plt.xlabel('Кількість сусідів (k)')
plt.ylabel('Середнє значення метрики')
plt.grid(True)
plt.show()


# Крок 7
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print(f'Класифікаційні метрики для тренувальної вибірки:\n{classification_report(y_train, y_train_pred)}')
print(f'Класифікаційні метрики для тестової вибірки:\n{classification_report(y_test, y_test_pred)}')

conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

print(f'Матриця змішування для тренувальної вибірки:\n{conf_matrix_train}')
print(f'Матриця змішування для тестової вибірки:\n{conf_matrix_test}')




# Візуалізація результатів на тестовій вибірці
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_test_pred_encoded = label_encoder.transform(y_test_pred)

plt.figure(figsize=(10, 6))
plt.title('Модель на тестовій вибірці')
plt.scatter(X_test.iloc[:, 8], X_test.iloc[:, 9], c=y_test_pred_encoded, cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


#Крок 8
leaf_sizes = range(20, 205, 5)
train_accuracies = []
test_accuracies = []

for leaf_size in leaf_sizes:
    knn_kd = KNeighborsClassifier(n_neighbors=optimal_k, leaf_size=leaf_size, algorithm='kd_tree')
    knn_kd.fit(X_train, y_train)
    train_accuracies.append(knn_kd.score(X_train, y_train))
    test_accuracies.append(knn_kd.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(leaf_sizes, train_accuracies, label='Train Accuracy')
plt.plot(leaf_sizes, test_accuracies, label='Test Accuracy')
plt.title('Вплив розміру листа на збалансовану точність для KNeighborsClassifier')
plt.xlabel('Розмір листа')
plt.ylabel('Збалансована точність')
plt.legend()
plt.grid(True)
plt.show()









# # Крок 7
# y_train_pred = knn.predict(X_train)
# y_test_pred = knn.predict(X_test)

# balanced_acc_train = balanced_accuracy_score(y_train, y_train_pred)
# roc_auc_train = roc_auc_score(y_train, knn.predict_proba(X_train), multi_class='ovr')

# balanced_acc_test = balanced_accuracy_score(y_test, y_test_pred)
# roc_auc_test = roc_auc_score(y_test, knn.predict_proba(X_test), multi_class='ovr')

# print("Класифікаційні метрики для тренувальної вибірки:")
# print(f"Збалансована точність: {balanced_acc_train}")
# print(f"ROC AUC: {roc_auc_train}")

# print("\nКласифікаційні метрики для тестової вибірки:")
# print(f"Збалансована точність: {balanced_acc_test}")
# print(f"ROC AUC: {roc_auc_test}")


# # Графічне представлення результатів на тестовій вибірці
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_test_pred, alpha=0.7)
# plt.title('Результати роботи моделі на тестовій вибірці')
# plt.xlabel('Справжні значення')
# plt.ylabel('Передбачені значення')
# plt.grid(True)
# plt.show()

# # Гістограма
# classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'X', 'Y']

# test_class_counts = [np.sum(y_test == cls) for cls in classes]
# predicted_class_counts = [np.sum(y_test_pred == cls) for cls in classes]

# plt.figure(figsize=(10, 6))
# plt.bar(classes, test_class_counts, color='blue', alpha=0.5, label='Тестова вибірка')
# plt.bar(classes, predicted_class_counts, color='orange', alpha=0.5, label='Передбачення моделі')
# plt.xlabel('Клас')
# plt.ylabel('Кількість об\'єктів')
# plt.title('Кількість об\'єктів в кожному класі для тестової вибірки та передбачених моделлю')
# plt.legend()
# plt.grid(True)
# plt.show()