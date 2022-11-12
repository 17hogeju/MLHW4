import numpy as np
import csv
import matplotlib.pyplot as plt

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
    
    def incremental_update_weights(self, lrate):
        epoch_error = 0
        for i in range(self.train.shape[0]):
            row_x = self.train[[i],:-1]
            row_y = self.train[i,-1]
            row_y = row_y.reshape((1,1))
            sign = np.dot(row_x, self.weights)
            sign = np.where(sign > 0, 1, -1)
            diff = np.subtract(row_y, sign)
            diff = diff * lrate
            delta_weights = np.dot(np.transpose(row_x), diff)
            self.weights = self.weights + delta_weights
            if (row_y * sign) < 0:
                epoch_error += 1
        return (epoch_error/self.train.shape[0])

    def batch_update_weights(self, lrate):
        x = self.train[:,:-1]
        y = self.train[:,[-1]]
        sign = np.dot(x, self.weights)
        sign = np.where(sign > 0, 1, -1)
        diff = np.subtract(y, sign)
        diff = diff * lrate
        delta_weights = np.dot(np.transpose(x), diff)
        self.weights = self.weights + delta_weights
        
        epoch_error = 0
        for i in range(self.train.shape[0]):
            if (int(y[i]) * int(sign[i])) < 0:
                epoch_error += 1

        return(epoch_error/self.train.shape[0])

    # def mse(self):
    #     prediction = np.dot(self.test[:,:-1], self.weights)
    #     diff = np.subtract(self.test[:,[-1]], prediction)
    #     return np.sum(diff**2)



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

def plotError(err, type):
    fig = plt.figure()
    plt.plot(range(EPOCHS), err)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    # plt.xticks(range(NUM_SPLITS))
    plt.title('Error during ' + type + ' training')
    #plt.legend()
    fig.savefig("figures/"+type+"/error_plot.png")
    plt.clf()

def plotDecSurf(epoch, weights, train, type):
    pred = np.dot(train[:,:-1], weights)
    pred = np.where(pred > 0, 1, -1)

    pos = np.where(pred > 0)
    neg = np.where(pred < 0)
    x1_max = int(np.ceil(np.amax(train[:, 1])))
    x1_min = int(np.ceil(np.amin(train[:, 1])))
    x2_max = int(np.ceil(np.amax(train[:, 2])))
    x2_min = int(np.ceil(np.amin(train[:, 2])))
    
    fig = plt.figure()
    plt.scatter(train[pos, 1], train[pos, 2], c ="r")
    plt.scatter(train[neg, 1], train[neg, 2], c ="b")
    plt.plot([x1_min, x1_max], [(2-x1_min)/3, (2-x1_max)/3],c="k")
    plt.axvline(x=0, c="lightgray", linestyle="dashed", label="x=0")
    plt.axhline(y=0, c="lightgray", linestyle="dashed", label="y=0")
    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Surface After Epoch = '+ str(epoch) + "("+type+")")
    fig.savefig("figures/"+type+"/dec_surf"+str(epoch)+".png")
    plt.clf()
    

def main():
    print("Assignment 3: Naive Bayes")
    generate_data(NUM_TRAIN+NUM_TEST)
    train_data = np.genfromtxt(TRAIN_DATA_PATH, delimiter=',')
    test_data = np.genfromtxt(TEST_DATA_PATH, delimiter=',')

    weights = np.random.uniform(-10, 10, size=(3, 1))

    err_inc = []
    percep_inc = Perceptron(train_data, test_data, weights)
    for epoch in range(EPOCHS):
        err_inc.append(percep_inc.incremental_update_weights(0.1))
        if (epoch == 4 or epoch == 9 or epoch == 49 or epoch == 99):
            plotDecSurf(epoch, percep_inc.weights, percep_inc.train, "incremental")

    plotError(err_inc, "incremental")

    err_batch = []
    percep_batch = Perceptron(train_data, test_data, weights)
    for epoch in range(EPOCHS):
        err_batch.append(percep_batch.batch_update_weights(0.1))
        if (epoch == 4 or epoch == 9 or epoch == 49 or epoch == 99):
            plotDecSurf(epoch, percep_batch.weights, percep_batch.train, "batch")


    plotError(err_batch, "batch")


if __name__ == "__main__":
    main()