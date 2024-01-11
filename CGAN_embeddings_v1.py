import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from tqdm import tqdm
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
from numpy.random import randint
from numpy import zeros
from numpy import ones
from numpy.random import randn
from tensorflow.keras.optimizers import Adam
from sklearn.manifold import TSNE
import seaborn as sns

print(os.getcwd())

#%%
# paths
seen_folder_path = os.path.join(os.getcwd(), "embeddings/seen")
csv_path = os.path.join(os.getcwd(), "seen.csv")
tb = SummaryWriter(os.path.join(os.getcwd(), "logs"))
plot_path = "C:/Users/102765181/Desktop/CGAN_images"

#%%

#variables

latent_dim = 100
num_examples_to_generate = 20
gen_lr = 0.0003
disc_lr = 0.0001
classes = 6
embed = 1024
#%%

# dataset
dataset = []
labels_dict = {}
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

dataset = np.array(dataset)
np.random.shuffle(dataset)
print(dataset.shape)
print(dataset[0][0].shape)

#%%

# generator architecture

def define_generator(latent_dim, n_classes=classes):
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

g_model = define_generator(latent_dim, classes)
g_model.summary()
#%%

# discriminator architecture
def define_discriminator(in_shape=(1024), n_classes=classes):
    # label input
    in_label = layers.Input(shape=(1,))
    # embedding for categorical input
    li = layers.Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    li = layers.Dense(1024)(li)
    li = layers.Reshape((1024,))(li)
    # reshape to additional channel
    in_image = layers.Input(shape=in_shape)
    # concat label as a channel
    merge = layers.Concatenate()([in_image, li])
    # downsample
    fe = layers.Dense(1024, activation='relu')(merge)
    fe = layers.Dense(512, activation='relu')(fe)
    fe = layers.Dropout(0.4)(fe)
    fe = layers.Dense(128, activation='relu')(fe)
    fe = layers.Dense(32, activation='relu')(fe)
    fe = layers.Dropout(0.4)(fe)
    # output
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    # define model
    model = tf.keras.Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.00005, beta_1=0.5, beta_2 = 0.999)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


d_model = define_discriminator()
d_model.summary()
#%%

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = tf.keras.Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(lr=0.0005, beta_1=0.5, beta_2 = 0.999)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

gan_model = define_gan(g_model, d_model)
gan_model.summary()
#%%

emb_model = tf.keras.Sequential([
    # label input
    layers.Input(shape=(1024)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.4),
    # output
    layers.Dense(6, activation='softmax')])
    # define model

emb_model.load_weights('Models/combined_embedding_model.hdf5')

emb_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])
emb_model.summary() # prints model summary


#%%
# generate images
seed = tf.random.normal([num_examples_to_generate, latent_dim])

seed.dtype

def generate_tsne(model, epoch, test_input):
    temp = []
    for i in range(6):
      seed = tf.random.normal([20, latent_dim])
      seed.dtype
      labels = label_gen(i)
      gen = model([test_input, labels], training=False)
      for emb in gen:
        temp.append(emb)
    temp = np.array(temp)
    embeddings = temp.reshape(120, 1024)
    
    tsne = TSNE(n_components=2 ,perplexity = 30,random_state = 1).fit_transform(embeddings)

    exp_labels = []
    for i in range(6):
        for j in range(20):
            exp_labels.append(i)

    tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1], 'label': exp_labels})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=sns.color_palette("hls", 6), data=tsne_df, legend="full", ax=ax,s=120)

    # you can change the limit
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_title("Generated embeddings on epoch " + str(epoch + 1))
    filename = str(epoch) + ".png"
    plt.savefig(os.path.join(plot_path, filename))
    plt.show()
    accuracy = 0
    if(epoch%50 == 0):
        exp_labels = tf.one_hot(exp_labels, depth=6)
        loss, acc = emb_model.evaluate(embeddings, exp_labels)
        print("Accuracy: " + str(acc))
        accuracy = acc
    if (accuracy > 0.9):
        return True
    else:
        return False        
def label_gen(n):
    lab = n
    return tf.repeat(lab, [20], axis=None, name=None)
#%%

def generate_latent_points(latent_dim, n_samples, n_classes=classes):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)

    return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y
#%%
def train(g_model, d_model, dataset, latent_dim, n_epochs):
    batch_size = 110
    # manually enumerate epochs
    for i in range(n_epochs):
    # enumerate batches over the training set
        start = time.time()
        dlf = 0
        dlr = 0
        gl = 0
        daf = 0
        dar =0
        for j in range(int(2860/batch_size)):
            start_index = j * batch_size
            end_index = start_index + batch_size
            batch = dataset[start_index:end_index]
            batch_emb = np.array([item[0] for item in batch])
            batch_label = np.array([item[1] for item in batch])
            batch_emb = batch_emb.reshape((batch_size, 1024))
            # train on real
            d_model.trainable = True
            d_loss1, d_acc1 = d_model.train_on_batch([batch_emb, batch_label], ones((batch_size, 1)))
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, batch_size)
            # update discriminator model weights
            d_loss2, d_acc2 = d_model.train_on_batch([X_fake, batch_label], y_fake)
            [z_input, labels_input] = generate_latent_points(latent_dim, batch_size)
            # create inverted labels for the fake samples
            y_gan = ones((batch_size, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            dlr += d_loss1
            dlf += d_loss2
            gl += g_loss
            dar += d_acc1
            daf +=d_acc2
            
        if (i%10 == 0):
             found = generate_tsne(g_model,
                              i,
                              seed)
        if (i == 1500):
            g_model.save('C:/Users/102765181/Desktop/Models/embedding_cgan_model_1500.hdf5')
        print ('Time for epoch {} is {} sec'.format(i + 1, time.time()-start))
        tb.add_scalar("Disc Loss fake per epoch", dlf/int(2860/batch_size), i)
        tb.add_scalar("Disc Loss real per epoch", dlr/int(2860/batch_size), i)
        tb.add_scalar("Gen Loss per epoch", gl/int(2860/batch_size), i)
        tb.close()
        print('Avg disc loss fake : {0}'.format(dlf/int(2860/batch_size)))
        print('Avg disc loss real : {0}'.format(dlr/int(2860/batch_size)))
        print('Avg gen loss : {0}'.format(gl/int(2860/batch_size)))
        print('Avg disc acc fake: {0}'.format(daf/int(2860/batch_size)))
        print('Avg disc acc real: {0}'.format(dar/int(2860/batch_size)))
        print("======================================")
        if(found):
            g_model.save('C:/Users/102765181/Desktop/Models/embedding_model_90.hdf5')
            break
     
#%%
train(g_model, d_model, dataset, latent_dim,2000)

