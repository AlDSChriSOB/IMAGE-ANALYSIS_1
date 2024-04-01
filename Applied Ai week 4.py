#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# In[2]:


data_dir = 'D:\AAI\mc-fakes-smaller'


# In[3]:


batch_size = 32
img_height = 160
img_width = 160
IMG_SIZE = (160, 160)




# In[4]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)


# In[5]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
         data_dir,
         validation_split=0.2,
         subset="validation",
         seed=123,
         image_size=(img_height, img_width),
         batch_size=batch_size)


# In[6]:


test_dataset = val_ds.take(5)
val_ds = val_ds.skip(5) 


# In[7]:


class_names = train_ds.class_names
print(class_names)


# In[8]:


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)


# In[9]:


# MLP

model = tf.keras.Sequential([
 tf.keras.layers.Flatten(input_shape=(160, 160, 3)),
 tf.keras.layers.Dense(128, activation='relu'),
 tf.keras.layers.Dense(2)
])

model.summary()


# In[10]:


model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy'])


# In[11]:


model.fit(train_ds, epochs=10, validation_data=val_ds)


# In[12]:


test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc) 


# In[13]:


# CNN

cnnmodel = tf.keras.Sequential()
cnnmodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
cnnmodel.add(layers.MaxPooling2D((2, 2)))
cnnmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnnmodel.add(layers.MaxPooling2D((2, 2)))
cnnmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnnmodel.add(layers.Flatten())
cnnmodel.add(layers.Dense(64, activation='relu'))
cnnmodel.add(layers.Dense(2))

cnnmodel.summary()


# In[14]:


cnnmodel.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy'])


# In[15]:


cnnmodel.fit(train_ds, epochs=10, validation_data=val_ds)


# In[16]:


test_loss, test_acc = cnnmodel.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc) 


# In[17]:


get_ipython().system('pip install tf-explain')


# In[22]:


# Generate visualisations


from tf_explain.core.grad_cam import GradCAM

IMAGE_PATH = "D:/AAI/mc-fakes-smaller/real/mc-48.jpg"

if __name__ == "__main__":
    model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()
    data = ([img], None)

    tabby_cat_class_index = 281
    explainer = GradCAM()
    # Compute GradCAM on VGG16
    grid = explainer.explain(
        data, model, class_index=tabby_cat_class_index, layer_name="block5_conv3"
    )
    explainer.save(grid, ".", "real_pic.png")


# In[ ]:




