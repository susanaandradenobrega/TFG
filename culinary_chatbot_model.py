import nltk
import numpy as np
import json
import pickle
import random 

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

# Loading the intents file(intents.json)
intents = json.loads(open("intents.json").read())

'''
Preprocess data
'''
# Generate the necessary structures that will be necesary 
words = []
classes = []
documents = []
ignore_punctuation_marks = ["?", "!", ".", ","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Separating words from patterns and add them to the list_words
        list_words = nltk.word_tokenize(pattern)
        words.extend(list_words)
        
        # Associating patterns with the corresponding tags
        documents.append(((list_words), intent['tag']))

        # Appending the tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Storing the root words or lemma 
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_punctuation_marks]
# Sort words and classes
words = sorted(set(words))
classes = sorted(classes)

# Saving the words and classes list to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create the training data with bag_of_words checking the word patterns
# present with a list of binary values and the corresponding tag as list of binary values
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag_of_words = []
    word_patterns = document[0]
    # Get the lemma for each word of word_patterns
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag_of_words.append(1) if word in word_patterns else bag_of_words.append(0)
          
    output_row = list(output_empty)
    # Put '1' for tag of the pattern and the others stay in '0' 
    output_row[classes.index(document[1])] = 1
    training.append([bag_of_words, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
  

# Split the data
# Patterns 
train_x = list(training[:, 0])
# Tags
train_y = list(training[:, 1])


# Create a Sequential machine learning model 3 layers. First layer 128 neurons, 
# second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile and fit the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y),epochs=200, batch_size=5, verbose=1)
  
# Saving the model as "chatbotModel.h5"
model.save("chatbotmodel.h5", hist)
