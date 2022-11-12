import numpy as np
import csv

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
NUM_TRAIN = 100
NUM_TEST = 50
RANGE = 1000
EPOCHS = 100
W0 = -2
W1 = 1
W2 = 3
X0 = 'x_0'
X1 = 'x_1'
X2 = 'x_2'
CLASS = 'class'


class Perceptron:
    def __init__(self, train: np.ndarray, test: np.ndarray, weights) -> None:
        self.train = train
        self.test = test
        self.weights = weights
    
    def batch_update_weights(self, lrate):
        epoch_error = 0
        for i in range(self.train.shape[0]):
            row_x = self.train[[i],:-1]
            row_y = self.train[i,-1]
            row_y = row_y.reshape((1,1))
            sign = np.dot(row_x, self.weights)
            sign = np.where(sign > 0, 1, -1)
            diff = np.subtract(row_y, sign)
            delta_weights = np.multiply((lrate * diff), row_x)
            delta_weights = np.transpose(delta_weights)
            self.weights = self.weights + delta_weights
            if (row_y * sign) < 0:
                epoch_error += 1
        return (epoch_error/self.train.shape[0])

    def mse(self):
        prediction = np.dot(self.test[:,:-1], self.weights)
        diff = np.subtract(self.test[:,[-1]], prediction)
        return np.sum(diff**2)



def generate_data(num_examples):
    print(f'Generating data in {TRAIN_DATA_PATH} and {TEST_DATA_PATH}')

    x0 = np.ones((num_examples, 1))
    x1 = np.random.uniform(-RANGE, RANGE, size=(num_examples, 1))
    x2 = np.random.uniform(-RANGE, RANGE, size=(num_examples, 1))
    x_data = np.concatenate((x0, x1, x2), axis=1)
    cl = np.dot(x_data, np.array([[W0],[W1],[W2]]))
    cl = np.where(cl > 0, 1, -1)
    data = np.concatenate((x_data, cl), axis=1)
    with open(TRAIN_DATA_PATH, 'w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerows(data[:NUM_TRAIN])
    with open(TEST_DATA_PATH, 'w', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerows(data[NUM_TRAIN:])

def main():
    print("Assignment 3: Naive Bayes")
    generate_data(NUM_TRAIN+NUM_TEST)
    train_data = np.genfromtxt(TRAIN_DATA_PATH, delimiter=',')
    test_data = np.genfromtxt(TEST_DATA_PATH, delimiter=',')

    err = []
    weights = np.random.uniform(-10, 10, size=(3, 1))

    percep = Perceptron(train_data, test_data, weights)
    for epoch in range(EPOCHS):
        err.append(percep.batch_update_weights(0.1))
        #print(percep.mse())
    print(percep.weights)
    print(err)


if __name__ == "__main__":
    main()