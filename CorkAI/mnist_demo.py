import tensorflow as tf

# TF 1.3 imports
from tensorflow.contrib import keras

# TF 1.11 imports
#from tensorflow import keras    



(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

        
"""
Problem constraint: the layer before the output (marked representation layer) must have only 2 (two) nodes.
This usually yields an accuracy of ~ 11.5% with eLU, and 41-43% with tanh on the representation layer.
"""
model = keras.models.Sequential([
    keras.layers.Reshape(input_shape = (28, 28), target_shape = (28, 28, 1)),
    keras.layers.Conv2D(filters = 16, kernel_size=(5,5), activation = tf.nn.elu),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.elu),    
    keras.layers.Dense(64, activation=tf.nn.elu),
    keras.layers.Dense(2, activation=tf.nn.tanh), # representation layer        
    keras.layers.Dense(10, activation=tf.nn.softmax)  
])



model.compile(optimizer=keras.optimizers.Adam(clipnorm = 1.), 
              loss='sparse_categorical_crossentropy',              
              metrics=['accuracy'],              
              )

model.fit(train_images, train_labels, epochs=300, batch_size = 10000)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

