import matplotlib.pyplot as plt
from keras import losses
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from numpy.random import seed
from sklearn.preprocessing import StandardScaler


def main():
    seed(1)

    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    x_train = StandardScaler().fit_transform(x_train)

    x_test = StandardScaler().fit_transform(x_test)

    neural_network_mnist1 = Sequential()
    neural_network_mnist1.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
    neural_network_mnist1.add(Dropout(0.01))

    neural_network_mnist1.add(Dense(5, activation='relu'))
    neural_network_mnist1.add(Dropout(0.01))

    neural_network_mnist1.add(Dense(1, activation='linear'))

    neural_network_mnist1.summary()

    sgd = SGD(lr=0.002)

    neural_network_mnist1.compile(optimizer=sgd, loss=losses.mean_squared_error
                                  )

    run_hist_1 = neural_network_mnist1.fit(x_train, y_train, epochs=500, \
                                           validation_data=(x_test, y_test), \
 \
                                           verbose=True, shuffle=False)

    print("Training neural network with dropouts..\n")
    print("Model evaluation Train data [loss]: ", neural_network_mnist1.evaluate(x_train, y_train))
    print("Model evaluation  Test Data [loss]: ", neural_network_mnist1.evaluate(x_test, y_test))

    neural_network_mnist = Sequential()
    neural_network_mnist.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
    neural_network_mnist.add(Dense(5, activation='relu'))

    neural_network_mnist.add(Dense(1, activation='linear'))

    neural_network_mnist.summary()

    sgd = SGD(lr=0.002)

    neural_network_mnist.compile(optimizer=sgd, loss=losses.mean_squared_error
                                 )

    run_hist_1 = neural_network_mnist.fit(x_train, y_train, epochs=500, \
                                          validation_data=(x_test, y_test), \
 \
                                          verbose=True, shuffle=False)

    print("Training neural network without dropouts..\n")
    print("Model evaluation Train data [loss]: ", neural_network_mnist.evaluate(x_train, y_train))
    print("Model evaluation  Test Data [loss]: ", neural_network_mnist.evaluate(x_test, y_test))
    plt.plot(run_hist_1.history["loss"], 'r', marker='.', label="Train Loss")
    plt.plot(run_hist_1.history["val_loss"], 'b', marker='.', label="Validation Loss")
    plt.title("Train loss and validation error with dropouts")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
