#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training


# In[3]:


train_images.shape


# In[4]:


train_images[0,23,23]  # let's have a look at one pixel


# In[5]:


train_labels[:10]  # let's have a look at the first 10 training labels


# In[6]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[7]:


plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()


# In[8]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[9]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])


# In[10]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[11]:


model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!


# In[12]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

print('Test accuracy:', test_acc)


# In[18]:


predictions = model.predict(test_images)
print(predictions[0])


# In[15]:


np.argmax(predictions[0])


# In[33]:


x=train_images[100]


# In[34]:


plt.figure()
plt.imshow(x)
plt.colorbar()
plt.grid(False)
plt.show()


# In[36]:


class_names[np.argmax(model.predict(np.array([x])))]


# In[ ]:




