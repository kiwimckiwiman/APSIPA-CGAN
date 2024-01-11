import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from tqdm import tqdm
import cv2
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
print(os.getcwd())

#%%
# paths
seen_folder_path = os.path.join(os.getcwd(), "plantvillagesmall/seen")
csv_path = os.path.join(os.getcwd(), "seen.csv")
tb = SummaryWriter(os.path.join(os.getcwd(), "logs"))
plot_path = "C:/Users/102765181/Desktop/CGAN_images"

#%%

#variables

latent_dim = 100
num_examples_to_generate = 25
gen_lr = 0.0003
disc_lr = 0.0001
classes = 6

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
  images = os.listdir(curr_folder_path)
  labels_dict[row[3]] = str(row[0])
  for i in tqdm(range(0,500)):
      select = random.choice(images)
      image = cv2.imread(os.path.join(curr_folder_path, select))
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image_array = tf.image.resize(np.array(image2), (256, 256))
      input_image = tf.expand_dims(image_array, axis=0)
      dataset.append((input_image, row[3]))
      images.remove(select)
      j += 1
      if len(images) == 0:
          break
      
print(len(dataset))
print(labels_dict)

dataset = np.array(dataset)
np.random.shuffle(dataset)
print(dataset.shape)
 
@tf.function
def normalization(tensor):
    #normalized_ds = data.map(lambda x: normalization_layer(x))
    tensor = tf.image.resize(
    tensor, (128,128))
    tensor = tf.subtract(tf.divide(tensor, 127.5), 1)
    return tensor


#%%

# generator architecture

def define_generator():
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,))

    label_embedding = layers.Flatten()(layers.Embedding(classes, latent_dim)(label))
    model_input = layers.Concatenate()([noise, label_embedding])

    x = layers.Dense(512 * 4 * 4)(model_input)
    
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.1)(x)
    
    x = layers.Conv2DTranspose(64 * 8, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(), name='conv_transpose_1')(x)
    x = layers.BatchNormalization(momentum=0.1, name='bn_1')(x)
    x = layers.LeakyReLU(name='leaky_relu_1')(x)
    
    x = layers.Conv2DTranspose(64 * 4, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(), name='conv_transpose_2')(x)
    x = layers.BatchNormalization(momentum=0.1, name='bn_2')(x)
    x = layers.LeakyReLU(name='leaky_relu_2')(x)
    
    x = layers.Conv2DTranspose(64 * 2, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(), name='conv_transpose_3')(x)
    x = layers.BatchNormalization(momentum=0.1, name='bn_3')(x)
    x = layers.LeakyReLU(name='leaky_relu_3')(x)

    x = layers.Conv2DTranspose(64 * 1, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(), name='conv_transpose_4')(x)
    x = layers.BatchNormalization(momentum=0.1, name='bn_4')(x)
    x = layers.LeakyReLU(name='leaky_relu_4')(x) 
    
    out_layer = layers.Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(),  activation='tanh', name='conv_transpose_6')(x)
    
   # define model
    model = tf.keras.Model([noise, label], out_layer)
    return model

conditional_gen = define_generator()

print(conditional_gen.summary())

#%%

# discriminator architecture

def label_condition_disc(in_shape=(128, 128,3), n_classes=classes, embedding_dim=100):
    # label input
    con_label = layers.Input(shape=(1,))
    # embedding for categorical input
    label_embedding = layers.Embedding(n_classes, embedding_dim)(con_label)
    # scale up to image dimensions with linear activation
    nodes = in_shape[0] * in_shape[1] * in_shape[2]
    label_dense = layers.Dense(nodes)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = layers.Reshape((in_shape[0], in_shape[1], 3))(label_dense)
    # image input
    return con_label, label_reshape_layer


def define_discriminator():
    con_label, label_condition_output = label_condition_disc()
    inp_image_output = layers.Input(shape=(128, 128,3))
    # concat label as a channel
    merge = layers.Concatenate()([inp_image_output, label_condition_output])
    
    x = layers.Conv2D(64, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(), name='conv_1')(merge)
    x = layers.LeakyReLU(0.2, name='leaky_relu_1')(x)
    
    x = layers.Conv2D(64 * 2, kernel_size=4, strides= 3, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal() ,name='conv_2')(x)
    x = layers.BatchNormalization(momentum=0.1, name='bn_1')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_2')(x)
    
    x = layers.Conv2D(64 * 2, kernel_size=4, strides= 3, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal() ,name='conv_3')(x)
    x = layers.BatchNormalization(momentum=0.1, name='bn_2')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_3')(x)
      
    x = layers.Conv2D(64 * 2, kernel_size=4, strides= 3, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal() ,name='conv_4')(x)
    x = layers.BatchNormalization(momentum=0, name='bn_3')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_4')(x)
  
 
    flattened_out = layers.Flatten()(x)
    # dropout
    dropout = layers.Dropout(0.4)(flattened_out)
    # output
    dense_out = layers.Dense(1, activation='sigmoid')(dropout)
    # define model


    # define model
    model = tf.keras.Model([inp_image_output, con_label], dense_out)
    return model

conditional_discriminator = define_discriminator()

print(conditional_discriminator.summary())

#%%

# loss + optim

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(label, fake_output):
    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss

def discriminator_loss(label, output):
    disc_loss = binary_cross_entropy(label, output)
    return disc_loss

generator_optimizer = tf.keras.optimizers.Adam(lr = gen_lr, beta_1 = 0.5, beta_2 = 0.999 )
discriminator_optimizer = tf.keras.optimizers.Adam(lr = disc_lr, beta_1 = 0.5, beta_2 = 0.999 )

#%%

# generate images
seed = tf.random.normal([num_examples_to_generate, latent_dim])

seed.dtype

def generate_and_save_images(model, epoch, test_input):
    labels = label_gen(n_classes=classes)
    predictions = model([test_input, labels], training=False)
    print(predictions.shape)
    label = str(labels_dict[np.array(labels)[0]])
    print("Generated Images are Conditioned on Label:", label)
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        pred = (predictions[i, :, :, :] + 1 ) * 127.5
        pred = np.array(pred)  
        plt.imshow(pred.astype(np.uint8))
        plt.axis('off')

    filename = label + "_epoch_" + str(epoch) + ".png"
    plt.savefig(os.path.join(plot_path, filename))
    plt.show()
    
def label_gen(n_classes):
    lab = tf.random.uniform((1,), minval=0, maxval=n_classes, dtype=tf.dtypes.int32, seed=None, name=None)
    return tf.repeat(lab, [25], axis=None, name=None)

#%%

# train step loop
@tf.function
def train_step(images, target, dlf, dlr, gl):
    # noise vector sampled from normal distribution
    noise = tf.random.normal([target.shape[0], latent_dim])
    # Train Discriminator with real labels
    with tf.GradientTape() as disc_tape1:
        generated_images = conditional_gen([noise,target], training=True)
        
        real_output = conditional_discriminator([images,target], training=True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)
        dlf += disc_loss1
    # gradient calculation for discriminator for real labels    
    gradients_of_disc1 = disc_tape1.gradient(disc_loss1, conditional_discriminator.trainable_variables)
    
    # parameters optimization for discriminator for real labels   
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc1,\
    conditional_discriminator.trainable_variables))
    
    # Train Discriminator with fake labels
    with tf.GradientTape() as disc_tape2:
        fake_output = conditional_discriminator([generated_images,target], training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)
        dlr += disc_loss2
    # gradient calculation for discriminator for fake labels 
    gradients_of_disc2 = disc_tape2.gradient(disc_loss2, conditional_discriminator.trainable_variables)
    
    
    # parameters optimization for discriminator for fake labels        
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc2,\
    conditional_discriminator.trainable_variables))
    
    # Train Generator with real labels
    with tf.GradientTape() as gen_tape:
        generated_images = conditional_gen([noise,target], training=True)
        fake_output = conditional_discriminator([generated_images,target], training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)
        gl += gen_loss
    # gradient calculation for generator for real labels     
        gradients_of_gen = gen_tape.gradient(gen_loss, conditional_gen.trainable_variables)
    
    # parameters optimization for generator for real labels
    generator_optimizer.apply_gradients(zip(gradients_of_gen,\
    conditional_gen.trainable_variables))    
    return dlf, dlr, gl

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        i = 0
        batch_size = 143
        dlf = 0
        dlr = 0
        gl = 0
        for j in range(int(len(dataset)/batch_size)):
            start_index = j * batch_size
            end_index = start_index + batch_size
            batch = dataset[start_index:end_index]
            images = np.array([item[0] for item in batch])
            labels = np.array([item[1] for item in batch])
            i += 1
            img = tf.cast(images, tf.float32)
            imgs = []
            for img, _ in batch:
                image = tf.cast(img, tf.float32)
                imgs.append(normalization(image))
            imgs = np.array(imgs)
            imgs = imgs.reshape(batch_size, 128, 128, 3)
            disc_loss_fake, disc_loss_real, gen_loss = train_step(imgs, labels, dlf, dlr, gl)
            dlf += disc_loss_fake
            dlr += disc_loss_real
            gl += gen_loss
        print(epoch)
        display.clear_output(wait=True)
        generate_and_save_images(conditional_gen,
                              epoch + 1,
                              seed)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        dlf = dlf.numpy()
        dlr = dlr.numpy()
        gl = gl.numpy()
        tb.add_scalar("Disc Loss fake per epoch", dlf / ((len(dataset)//batch_size) + (len(dataset) % batch_size > 0)), epoch)
        tb.add_scalar("Disc Loss real per epoch", dlr / ((len(dataset)//batch_size) + (len(dataset) % batch_size > 0)), epoch)
        tb.add_scalar("Gen Loss per epoch", gl / ((len(dataset)//batch_size) + (len(dataset) % batch_size > 0)), epoch)
        tb.close()
        
        dl1 = dlf / ((len(dataset)//batch_size) + (len(dataset) % batch_size > 0)), epoch
        dl2 = dlr / ((len(dataset)//batch_size) + (len(dataset) % batch_size > 0)), epoch
        gl_ = gl / ((len(dataset)//batch_size) + (len(dataset) % batch_size > 0)), epoch
        print('Total disc loss fake : {0}'.format(dl1))
        print('Total disc loss real : {0}'.format(dl2))
        print('Total gen loss : {0}'.format(gl_))
     
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(conditional_gen,
                            epochs,
                            seed)

train(dataset, 1500)
conditional_gen.save('C:/Users/102765181/Desktop/Models/image_cgan_model_1500.hdf5')
