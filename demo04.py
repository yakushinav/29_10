import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np

from collections import Counter
from sklearn.datasets import fetch_20newsgroups

#Категории текстов
categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]
#Обучающие и тестовые данные
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print('total texts in train:',len(newsgroups_train.data))
print('total texts in test:',len(newsgroups_test.data))

print('text',newsgroups_train.data[0])
print('category:',newsgroups_train.target[0])

#Словарь
vocab = Counter()
#Заполнение словаря
for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

print("Total words:", len(vocab))

total_words = len(vocab)

# Создание индексированного словаря
def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index

word2index = get_word_2_index(vocab)
print("Index of the word 'the':", word2index['the'])

#Векторизация текста: замена слов на их индексы в словаре
def text_to_vector(text):
    layer = np.zeros(total_words, dtype=float)
    for word in text.split(' '):
        layer[word2index[word.lower()]] += 1

    return layer

#Векторизация категории
def category_to_vector(category):
    y = np.zeros((3), dtype=float)
    if category == 0:
        y[0] = 1.
    elif category == 1:
        y[1] = 1.
    else:
        y[2] = 1.

    return y

#Формирование numPy массивов из текста
def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]

    for text in texts:
        layer = text_to_vector(text)
        batches.append(layer)

    for category in categories:
        y = category_to_vector(category)
        results.append(y)

    return np.array(batches), np.array(results)


print("Each batch has 100 texts and each matrix has 119930 elements (words):",   get_batch(newsgroups_train, 1, 100)[0].shape)
print("Each batch has 100 labels and each matrix has 3 elements (3 categories):",get_batch(newsgroups_train,1,100)[1].shape)

# Параметры
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Параметры нейронной сети
n_hidden_1 = 100  # Количество единиц в первом слое
n_hidden_2 = 100  # Количество единиц во втором слое
n_input = total_words  # Слов в словаре
n_classes = 3  # Категории: graphics, sci.space and baseball

#Входной и выходной тензор
input_tensor = tf.placeholder(tf.float32, [None, n_input], name="input")
output_tensor = tf.placeholder(tf.float32, [None, n_classes], name="output")

#Перцептрон
def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)

    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)

    # Output layer
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition


# Веса слоев
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Построение модели
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Обучение модели
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Инициализация
init = tf.global_variables_initializer()

# Сохранение обученной модели
saver = tf.train.Saver()

# Запуск графа вычислений
with tf.Session() as sess:
    sess.run(init)

    # Обучение в цикле
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train.data) / batch_size)
        # Проход по всем элементам
        for i in range(total_batch):
            batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
            # Запустим оптимизацию
            c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor: batch_y})
            # ошибка
            avg_cost += c / total_batch
        # вывод диагностических сообщений
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Проверка
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Расчет ошибки
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))

    # Сохраним обученную модель
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in path: %s" % save_path)



# Определение категории текста на обученной модели
text_for_prediction = newsgroups_test.data[5]

print('text',text_for_prediction)

print("Text correct category:", newsgroups_test.target[5])

# Векторизация
vector_txt = text_to_vector(text_for_prediction)
# Перевод в массив numPy
input_array = np.array([vector_txt])

input_array.shape

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    print("Model restored.")

    classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: input_array})
    print("Predicted category:", classification)

# Определение категории для 10 текстов

x_10_texts, y_10_correct_labels = get_batch(newsgroups_test, 0, 10)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    print("Model restored.")

    classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: x_10_texts})
    print("Predicted categories:", classification)
    print("Correct categories:", np.argmax(y_10_correct_labels, 1))