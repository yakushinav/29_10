#Подключение pandas
import pandas as pd
#sklearn библиотека для машинного обучения
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

#чтение данных
data = pd.read_csv('iris.csv')
#вывод набора данных
print(data.head(5))
#Разделяем выборку на обучающую и тестовую
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
#выведем матрицу корреляции признаков
corrmat = train.corr()
print("correlation", corrmat)
#поскольку модель машинного обучения работает только с признаками
#оставляем только значения values без имен столбцов
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']].values
y_train = train.species.values
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']].values
y_test = test.species.values

#модель классификатора на основе деревьев решений
mod_dt = tree.DecisionTreeClassifier(max_depth = 3, random_state = 1)
#обучение модели
mod_dt.fit(X_train,y_train)
#проверка на тестовой выборке
prediction=mod_dt.predict(X_test)
#качество модели
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,y_test))
#матрица ошибок
c_matrix=metrics.confusion_matrix(y_test, prediction)
print("confusion matrix\n", c_matrix)
#использование модели для предсказания
test_data=[[5.1,3.5,1.4,0.4]]
print("Prediction: ", mod_dt.predict(test_data))

#визуализация дерева в текстовом формате
text_representation = tree.export_text(mod_dt)
print(text_representation)


