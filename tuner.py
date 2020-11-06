from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense,Conv2D,Activation,Flatten,MaxPooling2D
from kerastuner.tuners import RandomSearch
import kerastuner.tuners
from kerastuner.engine.hyperparameters import HyperParameters
import time

LOG_DIR = f"{int(time.time())}"

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# plt.imshow(x_train[5],cmap="gray")
# plt.show()

def build_models(hp):
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int("input_units",min_value=32,max_value=256,step=32), (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range (hp.Int("n_layers",1,4)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units",min_value=32,max_value=256,step=32), (3, 3)))
        model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return model

# model = build_model()
# model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data = (x_test, y_test))
#

tuner = RandomSearch(
        build_models,
        objective = "val_accuracy",
        max_trials = 5,
        executions_per_trial =1,
        directory = LOG_DIR
        )
tuner.search(x=x_train,
             y=y_train,
             epochs=10,
             batch_size=64,
             validation_data=(x_test,y_test))

import pickle

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
     pickle.dump(tuner,f)

#
# tuner = pickle.load(open("tuner_1604492116.pkl","rb"))
# print(tuner.get_best_hyperparameters()[0].values)
# print(tuner.results_summary())    # gives top 10 best models
# print(tuner.get_best_models()[0].summary())   # can do direct .predict here as well




