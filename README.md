# AI chatbot

This chatbot is intergrated with Facebook, once its starts running the bot will fecth the unread thread in the chat windows of the user and will send the revelant response.

Packages used in the project
nltk
pickle
numpy
json
random
keras
sklearn

nltk:
This package is used for lemmatization of words present in the dataset. As this packages vast set of words which will provide accurate words by lemmatizing.

pickle:
This package is used for serilizae the nerual network, after training with the dataset the serialized file will be saved. While providing the response the bot will use this package to deserilizae and provide the response. This will eliminate the retraining bot for response.

numpy:
This package is used for spliting the trained result as the result are huge arrays. numpy can easily handle the huge datas. 

json:
This package is used to convert the dataset into JSON object. as the training dataset is in json syntax format.

random:
This is an inbuilt package present in the python. This is used to sent the response as the dataset mulitple response strings like (good morning, nice day etc)

keras:
This is the heart of the chatbot with this package we have created the neural network (3 layer network in the bot).

sklearn:
This package is used to show the plots for the results. (to show the accuray graphs and confusion matrix)
