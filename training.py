import json
import pickle
import random
import nltk
import numpy as np
import matplotlib.pyplot as plt


from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=2, validation_data=(np.array(train_x), np.array(train_y)))
model.save('chatbot_model.h5', hist)

y_pred = model.predict_classes(np.array(train_x))
print(y_pred)  #y->predicted
a = np.array(train_x)
test = model.predict_classes(a, batch_size=5)
print(test) #predicted classes
predicted = confusion_matrix(y_pred, test)
print(predicted)


fig, ax = plot_confusion_matrix(conf_mat=predicted, colorbar=True, class_names=classes)
plt.savefig("confusion.png")
plt.show()

model.summary()

loss_train = hist.history['loss'][0:199]
loss_val = hist.history['val_loss'][0:199]
epochs = range(1, 200)
plt.plot(epochs, loss_train, label='Training loss')
plt.plot(epochs, loss_val, label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Training_and_Validation_loss.png')
plt.show()

loss_train = hist.history['accuracy'][0:199]
loss_val = hist.history['val_accuracy'][0:199]
epochs = range(1, 200)
plt.plot(epochs, loss_train,  label='Training accuracy')
plt.plot(epochs, loss_val, label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Training_and_Validation_accuracy.png')
plt.show()