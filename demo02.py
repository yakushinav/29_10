import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras import layers


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Загрузка данных

# Здесь мы загружаем данные и преобразуем символьные метки в числовые

def read_data_bank():

    DF = pd.read_csv('bank-full.csv', delimiter=";", true_values=["success","yes"], false_values=["failure","no"])

    DF["education"] = pd.Categorical(DF["education"])

    DF["education"] = DF.education.cat.codes

    DF["marital"] = pd.Categorical(DF["marital"])

    DF["marital"] = DF.marital.cat.codes

    DF["job"] = pd.Categorical(DF["job"])

    DF["job"] = DF.job.cat.codes

    DF["contact"] = pd.Categorical(DF["contact"])

    DF["contact"] = DF.contact.cat.codes

    DF["month"] = pd.Categorical(DF["month"])

    DF["month"] = DF.month.cat.codes

    DF["poutcome"] = pd.Categorical(DF["poutcome"])

    DF["poutcome"] = DF.poutcome.cat.codes

    return DF

 

 

# Формирование набора данных, выделение признака target, обучающей train, тестовой test и контрольной val выборок

DF = read_data_bank()

 

target = DF.pop('y')

validation_size = 0.20

seed = 7

scoring = 'accuracy'

x_train, x_test, y_train, y_test = train_test_split(DF, target, test_size=validation_size, random_state=seed)

 

# Предобработаем данные (это массивы Numpy)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

 

y_train = y_train.astype('float32')

y_test = y_test.astype('float32')

 

# Зарезервируем 4000 примеров для валидации

x_val = x_train[-4000:]

y_val = y_train[-4000:]

x_train = x_train[:-4000]

y_train = y_train[:-4000]

 

#Создаем модель с заданной фукнцией ошибки и метрикой, обучаем модель и получаем оценку качества

def fit_model(loss_function, metric_function):

    inputs = keras.Input(shape=(16,), name='clients')

    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)

    x = layers.Dense(64, activation='relu', name='dense_2')(x)

    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

 

    model = keras.Model(inputs=inputs, outputs=outputs)

 

    # Укажем конфигурацию обучения (оптимизатор, функция потерь, метрики)

    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer

                  # Минимизируемая функция потерь

                  loss=loss_function,

                  metrics=[metric_function])

 

    # Обучим модель разбив данные на "пакеты"

    # размером "batch_size", и последовательно итерируя

    # весь датасет заданное количество "эпох"

    history = model.fit(x_train, y_train,

                        batch_size=64,

                        epochs=3,

                        # Мы передаем валидационные данные для

                        # мониторинга потерь и метрик на этих данных

                        # в конце каждой эпохи

                        validation_data=(x_val, y_val))

    results = model.evaluate(x_test, y_test, batch_size=128)

    predictions = model.predict(x_test[:5])

    return history, results, predictions

 

#Используем четыре разные функции ошибки, чтобы построить четыре разные модели

#И рисуем график качества

losses=['sparse_categorical_crossentropy','binary_crossentropy','mean_squared_error', 'mean_absolute_error']

metrics=['binary_accuracy','binary_accuracy','binary_accuracy','binary_accuracy']

 

binary_accuracy={}

 

for k in range(4):

    history, results, predictions=fit_model(losses[k],metrics[k])

    print(history.history)

    plt.plot(history.history['val_binary_accuracy'])

    binary_accuracy[losses[k]]=history.history['val_binary_accuracy'][-1]

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

 

plt.legend(losses, loc='upper left')

plt.show()

 

print(binary_accuracy)