import numpy as np
import cv2
import tensorflow as tf
import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from collections import Counter
from tensorflow.keras.applications import MobileNetV2
print(os.getcwd())
#%%
# paths
seen_folder_path = "plantvillagesmall/seen"
unseen_folder_path = "plantvillagesmall/unseen"
#%%
csv_path = "drive/MyDrive/Colab Notebooks/IIIP/Confirmed/seen.csv"

folders = os.listdir(seen_folder_path)
print(folders)

df = pd.read_csv(csv_path, header = None)
print(df)
#%%
# make train dataset
dataset = []
plant_labels = {}

#%%
for index,row in df.iterrows():
  curr_folder_path = seen_folder_path + "/" + str(row[0])
  j = 0
  images = os.listdir(curr_folder_path)
  plant_disease = row[0].split('___')
  plant_labels[row[1]] = str(plant_disease[0])
  for i in tqdm(range(0,500)):
      select = random.choice(images)
      image = cv2.imread(os.path.join(curr_folder_path, select))
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      dataset.append((image2, row[1]))
      images.remove(select)
      j += 1
      if len(images) == 0:
          break

print(len(dataset))
print(plant_labels)
#%%
np.random.shuffle(dataset)

train, test = train_test_split(dataset, test_size = 0.4)

train_images = np.array([item[0] for item in train])
train_labels = np.array([item[1] for item in train])

test_images = np.array([item[0] for item in test])
test_labels = np.array([item[1] for item in test])

print("Train length: ", str(len(train_images)))
print("Test length: ", str(len(test_images)))

#%%
# resize images, convert labels to one hot
train_images = tf.image.resize(train_images, (128, 128))
train_labels = tf.one_hot(train_labels, depth=3)

test_images = tf.image.resize(test_images, (128, 128))
test_labels = tf.one_hot(test_labels, depth=3)

#%%

image_input = tf.keras.layers.Input(shape=(128,128,3))
# call MobileNetV2 weights
mobilenet = MobileNetV2(include_top=False, input_tensor=image_input)

# set layers trainable to true
mobilenet.trainable = True

# set up model
neural_net = tf.keras.Sequential([
  tf.keras.layers.Input(shape = (128,128,3)),
  mobilenet,
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3,activation='softmax')
])

# compile model
neural_net.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
neural_net.summary() # prints model summary

# train the model
model_fit = neural_net.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# evaluate model
_, train_acc = neural_net.evaluate(train_images, train_labels)
_, test_acc = neural_net.evaluate(test_images, test_labels )

print(f'\nTrain accuracy: {train_acc:.0%}')
print(f'Test accuracy: {test_acc:.0%}')

neural_net.save('C:/Users/102765181/Desktop/Models/plant_fc_model.hdf5')

#%%
# make unseen dataset

for folder in os.listdir(unseen_folder_path):
  curr_folder_path = unseen_folder_path + "/" + folder
  unseen_dataset = []
  j = 0
  for i in tqdm(os.listdir(curr_folder_path)):
      image = cv2.imread(os.path.join(curr_folder_path, i))
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      unseen_dataset.append(image2)
      j += 1
  print(str(j) + " pictures found")
  unseen_dataset = np.array(unseen_dataset)
  unseen_dataset = tf.image.resize(unseen_dataset, (128, 128))
  print("Current label: " + folder)
  results = []
  predictions = neural_net.predict(unseen_dataset)
  for pred in predictions:
      results.append(plant_labels[np.argmax(pred)])
  freq = Counter(results)
  most_common = freq.most_common(1)[0]
  label = most_common[0]
  count = most_common[1]

  percentage = (count / len(results)) * 100
  print("Predicted Label: " + label + " | Percentage: " + str(percentage))
  print("=================================================")

#%%

# check with seen dataset

for folder in os.listdir(seen_folder_path):
  curr_folder_path = seen_folder_path + "/" + folder
  seen_dataset = []
  j = 0
  for i in tqdm(os.listdir(curr_folder_path)):
      image = cv2.imread(os.path.join(curr_folder_path, i))
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      seen_dataset.append(image2)
      j += 1
  print(str(j) + " pictures found")
  seen_dataset = np.array(seen_dataset)
  seen_dataset = tf.image.resize(seen_dataset, (128, 128))
  print("Current label: " + folder)
  results = []
  predictions = neural_net.predict(seen_dataset)
  for pred in predictions:
      results.append(plant_labels[np.argmax(pred)])
  freq = Counter(results)
  most_common = freq.most_common(1)[0]
  label = most_common[0]
  count = most_common[1]

  percentage = (count / len(results)) * 100
  print("Predicted Label: " + label + " | Percentage: " + str(percentage))
  print("=================================================")
#%%
folders = os.listdir(seen_folder_path)
print(folders)

df = pd.read_csv(csv_path, header = None)
print(df)
#%%
# make train dataset
dataset = []
disease_labels = {}

#%%
for index,row in df.iterrows():
  curr_folder_path = seen_folder_path + "/" + str(row[0])
  j = 0
  images = os.listdir(curr_folder_path)
  plant_disease = row[0].split('___')
  disease_labels[row[2]] = str(plant_disease[1])
  for i in tqdm(range(0,500)):
      select = random.choice(images)
      image = cv2.imread(os.path.join(curr_folder_path, select))
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      dataset.append((image2, row[2]))
      images.remove(select)
      j += 1
      if len(images) == 0:
          break

print(len(dataset))
print(disease_labels)
#%%
np.random.shuffle(dataset)

train, test = train_test_split(dataset, test_size = 0.4)

train_images = np.array([item[0] for item in train])
train_labels = np.array([item[1] for item in train])

test_images = np.array([item[0] for item in test])
test_labels = np.array([item[1] for item in test])

print("Train length: ", str(len(train_images)))
print("Test length: ", str(len(test_images)))

#%%
# resize images, convert labels to one hot
train_images = tf.image.resize(train_images, (128, 128))
train_labels = tf.one_hot(train_labels, depth=3)

test_images = tf.image.resize(test_images, (128, 128))
test_labels = tf.one_hot(test_labels, depth=3)

#%%

image_input = tf.keras.layers.Input(shape=(128,128,3))
# call MobileNetV2 weights
mobilenet = MobileNetV2(include_top=False, input_tensor=image_input)

# set layers trainable to true
mobilenet.trainable = True

# set up model
neural_net = tf.keras.Sequential([
  tf.keras.layers.Input(shape = (128,128,3)),
  mobilenet,
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3,activation='softmax')
])

# compile model
neural_net.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
neural_net.summary() # prints model summary

# train the model
model_fit = neural_net.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# evaluate model
_, train_acc = neural_net.evaluate(train_images, train_labels)
_, test_acc = neural_net.evaluate(test_images, test_labels )

print(f'\nTrain accuracy: {train_acc:.0%}')
print(f'Test accuracy: {test_acc:.0%}')

neural_net.save('C:/Users/102765181/Desktop/Models/disease_fc_model.hdf5')

#%%
# make unseen dataset

for folder in os.listdir(unseen_folder_path):
  curr_folder_path = unseen_folder_path + "/" + folder
  unseen_dataset = []
  j = 0
  for i in tqdm(os.listdir(curr_folder_path)):
      image = cv2.imread(os.path.join(curr_folder_path, i))
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      unseen_dataset.append(image2)
      j += 1
  print(str(j) + " pictures found")
  unseen_dataset = np.array(unseen_dataset)
  unseen_dataset = tf.image.resize(unseen_dataset, (128, 128))
  print("Current label: " + folder)
  results = []
  predictions = neural_net.predict(unseen_dataset)
  for pred in predictions:
      results.append(disease_labels[np.argmax(pred)])
  freq = Counter(results)
  most_common = freq.most_common(1)[0]
  label = most_common[0]
  count = most_common[1]

  percentage = (count / len(results)) * 100
  print("Predicted Label: " + label + " | Percentage: " + str(percentage))
  print("=================================================")

#%%

# check with seen dataset

for folder in os.listdir(seen_folder_path):
  curr_folder_path = seen_folder_path + "/" + folder
  seen_dataset = []
  j = 0
  for i in tqdm(os.listdir(curr_folder_path)):
      image = cv2.imread(os.path.join(curr_folder_path, i))
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      seen_dataset.append(image2)
      j += 1
  print(str(j) + " pictures found")
  seen_dataset = np.array(seen_dataset)
  seen_dataset = tf.image.resize(seen_dataset, (128, 128))
  print("Current label: " + folder)
  results = []
  predictions = neural_net.predict(seen_dataset)
  for pred in predictions:
      results.append(disease_labels[np.argmax(pred)])
  freq = Counter(results)
  most_common = freq.most_common(1)[0]
  label = most_common[0]
  count = most_common[1]

  percentage = (count / len(results)) * 100
  print("Predicted Label: " + label + " | Percentage: " + str(percentage))
  print("=================================================")
#%%
# paths

seen_embeddings = "embeddings_1024/seen"
unseen_embeddings = "embeddings_1024/unseen"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
import os
from tensorflow.keras.applications import MobileNetV2
#%%
image_input = tf.keras.layers.Input(shape=(128,128,3))
# call MobileNetV2 weights
mobilenet = MobileNetV2(include_top=False, input_tensor=image_input)

# set layers trainable to true
mobilenet.trainable = True

# set up model
plant_model = tf.keras.Sequential([
  tf.keras.layers.Input(shape = (128,128,3)),
  mobilenet,
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3,activation='softmax')
])


plant_model.load_weights('C:/Users/102765181/Desktop/Models/plant_fc_model.hdf5')

# compile model
plant_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# set up models
disease_model = tf.keras.Sequential([
  tf.keras.layers.Input(shape = (128,128,3)),
  mobilenet,
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3,activation='softmax')
])

disease_model.load_weights('C:/Users/102765181/Desktop/Models/disease_fc_model.hdf5')

# compile model
disease_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#%%
plant_model = tf.keras.models.Sequential(plant_model.layers[:-1])
disease_model = tf.keras.models.Sequential(disease_model.layers[:-1])

plant_model.summary()
disease_model.summary()

#%%
seen = "plantvillagesmall/seen"
unseen = "plantvillagesmall/unseen"

print(seen)
print(unseen)
#%%

for folder in os.listdir(seen):
  curr_folder_path = os.path.join(seen, folder)
  j = 0
  for filename in tqdm(os.listdir(curr_folder_path)):
   image = cv2.imread(os.path.join(curr_folder_path, filename))
   image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image_array = tf.image.resize(np.array(image2), (128, 128))
   input_image = tf.expand_dims(image_array, axis=0)

   seen_path = seen_embeddings + '/' + folder
   if not os.path.exists(seen_path):
       os.makedirs(seen_path)

   plant_embedding  = plant_model.predict(input_image)
   disease_embedding  = disease_model.predict(input_image)

   embedding = np.concatenate([plant_embedding, disease_embedding], axis=1)
   seen_path = os.path.join(seen_path, filename)
   np.save(seen_path, embedding)
   j += 1
  print(str(j) + " pictures converted")
#%%

for folder in os.listdir(unseen):
  curr_folder_path = os.path.join(unseen, folder)
  j = 0
  for filename in tqdm(os.listdir(curr_folder_path)):
   image = cv2.imread(os.path.join(curr_folder_path, filename))
   image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image_array = tf.image.resize(np.array(image2), (128, 128))
   input_image = tf.expand_dims(image_array, axis=0)

   seen_path = unseen_embeddings + '/' + folder
   if not os.path.exists(seen_path):
       os.makedirs(seen_path)

   plant_embedding  = plant_model.predict(input_image)
   disease_embedding  = disease_model.predict(input_image)

   embedding = np.concatenate([plant_embedding, disease_embedding], axis=1)
   seen_path = os.path.join(seen_path, filename)
   np.save(seen_path, embedding)
   j += 1
  print(str(j) + " pictures converted")
#%%
# make train dataset
dataset = []
plant_labels = {}

#%%
dataset = []
labels_dict = {}
seen_folder_path = "../../embeddings_1024/seen"
folders = os.listdir(seen_folder_path)
print(folders)

df = pd.read_csv(csv_path, header = None)
print(df)
      
for index,row in df.iterrows():
  curr_folder_path = seen_folder_path + "/" + str(row[0])
  j = 0
  embeddings = os.listdir(curr_folder_path)
  labels_dict[row[3]] = str(row[0])
  for i in tqdm(range(0,500)):
      select = random.choice(embeddings)
      data = np.load(os.path.join(curr_folder_path, select))
      dataset.append((data, row[3]))
      embeddings.remove(select)
      j += 1
      if len(embeddings) == 0:
          break
      
print(len(dataset))
print(labels_dict)
#%%
dataset = np.array(dataset)
print(dataset.shape)
#%%
np.random.shuffle(dataset)

train, test = train_test_split(dataset, test_size = 0.4)

train_images = np.array([item[0] for item in train])
train_labels = np.array([item[1] for item in train])

test_images = np.array([item[0] for item in test])
test_labels = np.array([item[1] for item in test])

print("Train length: ", str(len(train_images)))
print("Test length: ", str(len(test_images)))

print(train_images.shape)
#%%
# resize images, convert labels to one hot
train_images = train_images.reshape(len(train_images), 1024)
train_labels = tf.one_hot(train_labels, depth=6)

test_images = test_images.reshape(len(test_images), 1024)
test_labels = tf.one_hot(test_labels, depth=6)

#%%
print(train_images.shape)
#%%
embed_classifier = tf.keras.Sequential([
    # label input
    tf.keras.layers.Input(shape=(1024)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    # output
    tf.keras.layers.Dense(6, activation='softmax')])
    # define model
    
embed_classifier.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])
embed_classifier.summary() # prints model summary


# train the model
model_fit = embed_classifier.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))       

# evaluate model
_, train_acc = embed_classifier.evaluate(train_images, train_labels)
_, test_acc = embed_classifier.evaluate(test_images, test_labels )

print(f'\nTrain accuracy: {train_acc:.0%}')
print(f'Test accuracy: {test_acc:.0%}')

embed_classifier.save('C:/Users/102765181/Desktop/Models/embedding_classifier.hdf5')

#%%
embedding_classifier = 'C:/Users/102765181/Desktop/Models/embedding_classifier.hdf5'

embed_classifier = tf.keras.Sequential([
    # label input
    tf.keras.layers.Input(shape=(1024)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    # output
    tf.keras.layers.Dense(6, activation='softmax')])
    # define model

embed_classifier.load_weights(embedding_classifier)

embed_classifier.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])

# check with seen dataset
eval_set = []
for index,row in df.iterrows():
  curr_folder_path = "../../embeddings_1024/seen/" + str(row[0])
  j = 0
  images = os.listdir(curr_folder_path)
  for i in tqdm(os.listdir(curr_folder_path)):
      data = np.load(os.path.join(curr_folder_path, i))
      eval_set.append((data, row[3]))

np.random.shuffle(eval_set)
eval_images = np.array([item[0] for item in eval_set])
eval_labels = np.array([item[1] for item in eval_set])
eval_images = eval_images.reshape(len(eval_images), 1024)
eval_labels = tf.one_hot(eval_labels, depth=6)
loss, acc = embed_classifier.evaluate(eval_images, eval_labels)
print("Accuracy: "+ str(acc))