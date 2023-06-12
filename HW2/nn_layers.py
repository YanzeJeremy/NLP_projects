import numpy as np

from abc import ABC, abstractmethod
np.random.seed(42)
class NNComp(ABC):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your NN modules as concrete
    implementations of this class, and fill forward and backward
    methods for each module accordingly.
    """

    @abstractmethod
    def forward(self, x):
        raise NotImplemented

    @abstractmethod
    def backward(self, incoming_grad):
        raise NotImplemented

        
class FeedForwardNetwork(NNComp):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your FeedForwardNetwork as concrete
    implementations of the NNComp class, and fill forward and backward
    methods for each module accordingly. It will likely be composed of
    other NNComp objects.
    """
    def __init__(self,u,N_x,N_y):
        self.u = u
        self.N_x = N_x
        self.N_y = N_y

        self.W = np.random.randn(self.N_x, self.u)* np.sqrt(1/self.N_x)
        self.U = np.random.randn(self.u, self.N_y)* np.sqrt(1/self.N_x)
        self.B_1 = np.zeros((1, self.u))
        self.B_2 = np.zeros((1,self.N_y))

    def forward(self, x):
        """
        Forward pass for neural network using RELU as activation function.
        returns the X input,the input and output of different layers.
        """
        A1 = np.matmul(x, self.W)+self.B_1 #(10,5)
        H1 = self.relu(A1) #(10,5)
        A2 = np.matmul(H1, self.U) + self.B_2 #(10,4)
        H2 = np.array([self.softmax(ss) for ss in A2]) #(10,4)
        return x,A1,H1,A2,H2

    def backward(self,x,A1,H1,A2,H2,labels):
        """
        Backward pass for neural network. Also contains the true label reshape.
        returns the derivative of w1,w2,b1,b2
        """
        labels_shaped = np.zeros(np.shape(H2)) #(10,4)
        for index,i in enumerate(labels):
            labels_shaped[index][i] = 1
        grad_y_pred = H2 - labels_shaped #(10,4)
        delta2 = np.matmul(H1.T,grad_y_pred)
        db2 = grad_y_pred.mean(axis=0)
        grad_h = np.matmul(grad_y_pred,self.U.T)
        temp = np.multiply(grad_h, self.dev_relu(A1))
        delta1 = np.matmul(x.T,temp)
        db1 = temp.mean(axis=0)
        return delta1,delta2,db1,db2

    def relu(self,z):
        """
        RELU function
        """
        return np.maximum(z,0)

    def dev_relu(self,x):
        """
        Derivative of RELU function
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def softmax(self,x):
        """
        Softmax function
        """
        ex = np.exp(x - np.max(x))
        return ex / ex.sum()

    def loss(self,y_true,y_pred):
        """
        Cross-entropy loss function
        """
        pro = -np.log(y_pred)
        loss = [pro[index][i] for index,i in enumerate(y_true)]
        loss_mean = np.mean(loss)
        return loss_mean

