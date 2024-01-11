# import
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
print(os.getcwd())
#%%

embedding_generator = 'C:/Users/102765181/Desktop/Models/embedding_model_95.hdf5'
embeddings_seen = "../../embeddings_1024/seen"
embedding_classifier = 'C:/Users/102765181/Desktop/Models/embedding_classifier.hdf5'
result_images = "../../Results"
csv_path = "../../seen.csv"
df = pd.read_csv(csv_path, header = None)

#%%

latent_dim = 100
classes = 6
num_to_generate = 20

#%%

labels_dict = {0: 'Tomato___Bacterial_spot', 1: 'Peach___Bacterial_spot', 2: 'Peach___healthy', 3: 'Tomato___healthy', 4: 'Potato___Early_blight', 5: 'Tomato___Early_blight'}
eval_set = []
true_labels = []
real_embeddings = []
eval_labels_real = []
      
for index,row in df.iterrows():
  curr_folder_path = embeddings_seen + "/" + str(row[0])
  j = 0
  embeds = os.listdir(curr_folder_path)
  true_labels.append(row[3])
  for i in tqdm(range(num_to_generate)):
      data = np.load(os.path.join(curr_folder_path, embeds[i]))
      eval_set.append((data, row[3]))
      
real_embeddings = np.array([item[0] for item in eval_set])
eval_labels_real = np.array([item[1] for item in eval_set])

#%%

print(len(real_embeddings))
print(real_embeddings.shape)
real_embeddings = real_embeddings.reshape(((num_to_generate*classes),1024))
print(real_embeddings.shape)

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

embed_classifier.load_weights(embedding_classifier)

embed_classifier.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])

#%%

def define_generator(latent_dim=latent_dim, n_classes=classes):
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,))

    label_embedding = layers.Flatten()(layers.Embedding(classes, latent_dim)(label))
    model_input = layers.Concatenate()([noise, label_embedding])

    x = layers.Dense(64)(model_input)
    x = layers.Dense(256, activation = 'relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation = 'relu')(x)
    out_layer = layers.Dense(1024, activation = 'linear')(x)
        
   # define model
    model = tf.keras.Model([noise, label], out_layer)
    return model

#%%

model = define_generator()
model.load_weights(embedding_generator)

#%%

def label_gen(n):
    lab = n
    return tf.repeat(lab, [num_to_generate], axis=None, name=None)

#%%

fake_embeddings = []
for i in range(6):
  seed = tf.random.normal([num_to_generate, latent_dim])
  seed.dtype
  labels = label_gen(i)
  gen = model([seed, labels], training=False)
  for emb in gen:
    fake_embeddings.append(emb)
fake_embeddings = np.array(fake_embeddings).reshape((num_to_generate*classes), 1024)
#%%

temp = []
for i in range(6):
  seed = tf.random.normal([20, latent_dim])
  seed.dtype
  labels = label_gen(i)
  gen = model([seed, labels], training=False)
  for emb in gen:
    temp.append(emb)
temp = np.array(temp)

embeddings = temp.reshape(120, 1024)

exp_labels = []
for i in range(6):
    for j in range(20):
        exp_labels.append(i)
        
exp_labels = tf.one_hot(exp_labels, depth=6)
loss, acc = embed_classifier.evaluate(embeddings, exp_labels)

#%%
print(fake_embeddings.shape)

#%%

exp_labels = []
for i in true_labels:
    for j in range(20):
        exp_labels.append(i)

tsne = TSNE(n_components=2 ,perplexity = 30,random_state = 1).fit_transform(real_embeddings)

tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_labels})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 6), data=tsne_df, legend="full", ax=ax,s=120)
tsne_df_new = tsne_df.to_numpy()
lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

# you can change the limit
ax.set_xlim(-30,30)
ax.set_ylim(-30,30)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax.set_title("Real_embeddings")
filename = result_images+"/Real_embeddings_tsne.png"
plt.savefig(filename)
plt.show()
#%%

tsne = TSNE(n_components=2 ,perplexity = 50,random_state = 1).fit_transform(fake_embeddings)

tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_labels})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 6), data=tsne_df, legend="full", ax=ax,s=120)
tsne_df_new = tsne_df.to_numpy()
lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

# you can change the limit
ax.set_xlim(-7.5,2.5)
ax.set_ylim(-9,2)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax.set_title("Fake_embeddings")
filename = result_images+"/Fake_embeddings_tsne.png"
plt.savefig(filename)
plt.show()
#%%
for n in range(1, 7):
    real_fake_labels = [0,1]
    exp_rf_labels = []
    for i in real_fake_labels:
        for j in range(20):
            exp_rf_labels.append(i)
    
    rf_embeddings = []
    for i in range((n*20)-20, (n*20)):
        rf_embeddings.append(real_embeddings[i])
        
    for i in range((n*20)-20, (n*20)):
        rf_embeddings.append(fake_embeddings[i])
        
    tsne = TSNE(n_components=2 ,perplexity = 30,random_state = 1).fit_transform(rf_embeddings)
    tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_rf_labels})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 2), data=tsne_df, legend="full", ax=ax,s=120)
    tsne_df_new = tsne_df.to_numpy()
    lim = (tsne_df_new.min()-5, tsne_df_new.max()+5)

    # you can change the limit
    ax.set_xlim(-250,250)
    ax.set_ylim(-250,250)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_title("Class " + str(n-1) + ": " + labels_dict[n-1])
    filename = result_images+"/Real_v_fake_class_" + str(n-1) + ".png"
    plt.savefig(filename)
    plt.show()
#%%
fake_overall_labels = []

for n in range(1,7):
    fake_class_embeddings = []
    real_class_embeddings = []
    fake_labels = []
    real_labels = []
    for i in range((n*20)-20, (n*20)):
        fake_class_embeddings.append(fake_embeddings[i])
    for i in range((n*20)-20, (n*20)):
        real_class_embeddings.append(real_embeddings[i])
        real_labels.append(eval_labels_real[i])
    for i in range(20):
        fake_overall_labels.append(n-1)
        fake_labels.append(n-1)
    
    fake_class_embeddings = np.array(fake_class_embeddings).reshape(len(fake_class_embeddings), 1024)
    real_class_embeddings = np.array(real_class_embeddings).reshape(len(real_class_embeddings), 1024)

    fake_labels = tf.one_hot(np.array(fake_labels), depth=6)
    real_labels = tf.one_hot(np.array(real_labels), depth=6)


    print("=======================================")
    print("Class-wise Accuracy: Class " + str(n))
    loss, racc = embed_classifier.evaluate(real_class_embeddings, real_labels)
    loss, facc = embed_classifier.evaluate(fake_class_embeddings, fake_labels)
    print("Real class Accuracy: " + str(racc))
    print("Fake class Accuracy: " + str(facc))

print("=======================================")
print("Overall Accuracy:")
fake_overall_labels = tf.one_hot(np.array(fake_overall_labels), depth=6)
eval_labels_real = tf.one_hot(np.array(eval_labels_real), depth=6)

loss, racc = embed_classifier.evaluate(real_embeddings, eval_labels_real)
loss, facc = embed_classifier.evaluate(fake_embeddings, fake_overall_labels)
print("Real Overall Accuracy: " + str(racc))
print("Fake Overall Accuracy: " + str(facc))