import numpy as np
import csv

TRAIN_DATA_PATH = '../data/train.csv'
TEST_DATA_PATH = '../data/test.csv'
NUM_TRAIN = 1500
NUM_TEST = 500
RANGE = 1000
EPOCHS = 500
W0 = -2
W1 = 1
W2 = 3
X0 = 'x_0'
X1 = 'x_1'
X2 = 'x_2'
CLASS = 'class'


class Perceptron:
    def __init__(self) -> None:
        self.weights = np.array([[W0],[W1],[W2]])

    def batch_update_weights(self, train_x: np.ndarray, train_y: np.ndarray, lrate):
        for i in range(len(self.weights)):
            o = np.dot(self.weights, train_x)
            grad_desc = sum(np.subtract(train_y, o) * -(train_x[:,[i]]))
            delta_weight = -lrate * grad_desc
            self.weights[i] += delta_weight

    def mse(self, test_x: np.ndarray, test_y: np.ndarray):
        prediction = np.dot(self.weights, test_x)
        diff = np.subtract(test_y - prediction)
        return np.sum(diff**2)



def generate_data(num_examples):
    print(f'Generating data in {TRAIN_DATA_PATH} and {TEST_DATA_PATH}')

    x0 = np.ones((num_examples, 1))
    x1 = np.random.uniform(-RANGE, RANGE, size=(num_examples, 1))
    x2 = np.random.uniform(-RANGE, RANGE, size=(num_examples, 1))
    x_data = np.concatenate((x0, x1, x2), axis=1)
    cl = np.dot(x_data, np.array([[W0],[W1],[W2]]))
    data = np.concatenate((x_data, cl), axis=1)
    with open(TRAIN_DATA_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data[:NUM_TRAIN])
    with open(TEST_DATA_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data[NUM_TRAIN:])

def main():
    print("Assignment 3: Naive Bayes")
    generate_data(NUM_TRAIN+NUM_TEST)
    train_data = np.genfromtxt(TRAIN_DATA_PATH, delimiter=',')
    train_x = train_data[:,:-1]
    train_y = train_data[:, [-1]]

    test_data = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    test_x = test_data[:,:-1]
    test_y = test_data[:, [-1]]

    percep = Perceptron()
    # for epoch in range(EPOCHS):
    #     percep.batch_update_weights(train_x, train_y, .1)


                  

if __name__ == "__main__":
    main()