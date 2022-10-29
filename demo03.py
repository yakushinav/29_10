# Пример 3. Как работает машинное обучение
# Для работы нужны библиотеки pandas, sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Загрузка файла
data = pd.read_csv("data01.csv")

# Формирование набора данных, выделение признака target, обучающей train, тестовой test и контрольной val выборок
# Числовые данные
array = data.values
X = array[:, 0]
# Метка класификации
Y = array[:, 1]
# Размер проверочной выборки 20% от всех данных
validation_size = 0.20
#Указывает, что выбор случайных данных должен быть одинаковым при каждом вызове обучения
seed = 7
#Разделение данных на тренировочные и проверочные
#X_train, X_validation - тренировочные проверочные данные для числовых данных
#Y_train, Y_validation - тренировочные проверочные данные для метки
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,random_state=seed, shuffle=True)

scoring = 'accuracy'
#Кросс-валидация K-fold - это систематический процесс повторения процедуры разделения
# тренировочных / тестовых данных несколько раз, чтобы уменьшить дисперсию, связанную
# с одним разделением. Вы по существу разделяете весь набор данных на K равными размерами
# «складки», и каждая складка используется один раз для тестирования модели и
# K-1 раз для обучения модели.
# в нашем случае 10 раз
kfold = model_selection.KFold(n_splits=2, random_state=seed, shuffle=True)

#Теперь проверим полученные модели с помощью скользящего контроля.
# Для этого нам необходимо воcпользоваться функцией cross_val_score
cv_results = model_selection.cross_val_score(KNeighborsClassifier(), X_train.reshape(-1,1), Y_train, cv=kfold, scoring=scoring)
#Среднее значение и среднеквадратичное отклонение
msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())

#создаем модель K-ближайших соседей
knn = KNeighborsClassifier()
#обучаем модель
knn.fit(X_train.reshape(-1,1), Y_train)
#проверяем качество обученной модели на тестовых данных
predictions = knn.predict(X_validation.reshape(-1,1))

# Описание входных данных
print(data.describe())
#Группировка данных по метке классификации
print(data.groupby('target').size())
# Оценка качества модели
print(msg)
#Средняя ошибка распознавания
print(accuracy_score(Y_validation, predictions))
#Количество распознанных чисел по видам
print(confusion_matrix(Y_validation, predictions))
#сводная таблица распределения вероятностей распознавания чисел
print(classification_report(Y_validation, predictions))