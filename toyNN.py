import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.random.seed(10)

class Dense:
    def __init__(self, input_size, output_size, init_factor=0.01):
        self.W = init_factor * np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, doutput):
        self.dX = np.dot(doutput, self.W.T)
        self.dW = np.dot(self.X.T, doutput)
        self.db = np.sum(doutput, axis=0, keepdims=True)
        return self.dX

    def update(self, eta, reg):
        self.dW += reg * self.W
        self.W -= eta * self.dW
        self.b -= eta * self.db
        #print np.linalg.norm(self.dW), np.linalg.norm(self.db)

    def __repr__(self):
        return "Dense Layer %s" % str(self.W.shape)

class ReLU:
    def forward(self, X):
        self.X = X
        self.output = np.maximum(0, self.X)
        return self.output

    def backward(self, doutput):
        self.dX = doutput * (self.X > 0)
        return self.dX

    def update(self, eta, reg):
        pass

    def __repr__(self):
        return "ReLU Layer %s" % str(self.X.shape)

class CrossEntropy:
    def score(self, ypred, y):
        exp_scores = np.exp(ypred)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(len(y)), y])
        data_loss = np.sum(correct_logprobs) / len(y)
        return data_loss

    def dscore(self, ypred, y):
        exp_scores = np.exp(ypred)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  
        probs[range(len(y)), y] -= 1
        dscore = probs / len(y)
        return dscore

class NN:
    def __init__(self, nb_epoch, eta, n_unit, n_output, reg, early_stopping=False, graph=False):
        self.nb_epoch = nb_epoch
        self.eta = eta
        self.n_unit = n_unit
        self.reg = reg
        self.graph = graph
        if self.graph:
            self.Z = []
        self.early_stopping = early_stopping
        self.layers = [
                Dense(Xtr.shape[1], self.n_unit),
                ReLU(),
                #Dense(self.n_unit, self.n_unit),
                #ReLU(),
                Dense(self.n_unit, n_output)
                ]
        self.objective = CrossEntropy()

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y):
        doutput = y
        for layer in reversed(self.layers):
            doutput = layer.backward(doutput)
            layer.update(self.eta, self.reg)        

    def predict(self, X):
        return self.forward(X)

    def train(self, Xtr, ytr, Xva, yva):
        if self.early_stopping:
            lowest_score = np.inf
            past_steps = 0
        for n in range(self.nb_epoch):
            output = self.forward(Xtr)
            self.backward(self.objective.dscore(output, ytr))
            score_tr = self.objective.score(self.predict(Xtr), ytr)
            score_va = self.objective.score(self.predict(Xva), yva)
            if self.early_stopping:
                if score_va < lowest_score:
                    lowest_score = score_va
                    past_steps = 0
                else:
                    past_steps += 1
                    if past_steps > 100:
                        break
            if self.graph and n % 30 == 0:
                h = 0.02
                x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
                y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = np.argmax(Z, axis=1)
                Z = Z.reshape(xx.shape)
                self.Z.append(Z)

            print "epoch %d: train score: %f, validation score: %f" % (n, score_tr, score_va)

def generate_data():
    # from Stanford CS231n
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in xrange(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

def show_plot(Xtr, nn):
    h = 0.02
    x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
    y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(1, figsize=(4, 4))

    def animate(t):
        ax.contourf(xx, yy, nn.Z[t], cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        return ax,

    ani = animation.FuncAnimation(fig, animate, len(nn.Z), blit=False)
    plt.show()    

if __name__ == '__main__':
    X ,y = generate_data()
    Xtr = X[::2]
    ytr = y[::2]
    Xva = X[1::2]
    yva = y[1::2]

    nn = NN(nb_epoch=10000, eta=5e-1, n_unit=100, n_output=3, reg=1e-3, early_stopping=False, graph=True)
    nn.train(Xtr, ytr, Xva, yva)

    show_plot(Xtr, nn)

