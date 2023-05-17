import tensorflow as tf
# import deepxde as dde
import numpy as np
from scipy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.io
import time
import timeit
from mpi4py import MPI
# import tensorflow_probability as tfp

# tf.disable_v2_behavior()
print(tf.__version__)

class DeepvdP:
    # Initialize the class
    def __init__(self, x, t, layers):

        self.lb = t.min(0)
        self.ub = t.max(0)
        
        self.x = x
        self.t = t
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])

        self.x_control = 2 * tf.math.sin(self.t_tf)#5.00 *
        self.x_dd_control = - 2 * tf.math.sin(self.t_tf) #5.00 *

        self.x_pred, self.x_dd_pred, self.ICs = self.vdP(self.t_tf)

        self.x_res = self.x_control - self.x_pred
        self.x_dd_res = self.x_dd_control - self.x_dd_pred

        # Signal-encoded control
        self.control = self.x_res + self.x_dd_res

        self.loss = 1 * tf.reduce_sum(tf.square(self.control)) + tf.reduce_sum(tf.square(self.ICs)) + \
        tf.reduce_sum(tf.square( self.x_tf - (self.x_pred) )) #+ \* 2 | * 2 / 5
                    

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 200000,
                                                                           'maxfun': 200000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps}) 


        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
# ==============================================================================================
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
   
    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
# ==============================================================================================
    def vdP(self, t):
      x = self.neural_net(tf.concat([t],1), self.weights, self.biases)
      # dx_t = dde.grad.jacobian(x, t, i=0)
      # dx_tt = dde.grad.hessian(x, t, i=0)
      dx_t = tf.compat.v1.gradients(x, t)
      dx_tt = tf.compat.v1.gradients(dx_t, t)
      # x_desire = tf.math.sin(t) #5 * 
      # x_dot_desire = tf.math.cos(t) #5 * 
      # x_ddot_desire = - tf.math.sin(t) #-5 * 
      # control = (x_desire - x) + (x_ddot_desire - dx_tt)
      ICs = x[0] - 1# Initial condition: 1, 5, 10
      return x, dx_tt, ICs
# ==============================================================================================
    def callback(self, loss): #, betta
        print('%.3e' % (loss)) #, betta B: %.5f
        return loss
      
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.t_tf: self.t}

        # var_loss = tf.Variable(tf_dict)
        # loss = lambda: (var_loss ** 2)/2.0 

        self.sess.run(self.train_op_Adam, feed_dict = tf_dict)

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

        # self.optimizer.minimize(self.loss)
        # self.optimizer.minimize(self.loss, global_step=None, var_list=var_loss,
    # aggregation_method=None, colocate_gradients_with_ops=False, name=None,
    # grad_loss=None)

        # (self.sess, feed_dict = tf_dict, fetches = [self.loss], loss_callback = self.callback)
            
# ==============================================================================================    
    def predict(self, t_star):
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t}  
        x_star = self.sess.run(self.x_pred, tf_dict)
        
        return x_star

if __name__ == "__main__": 
    
    layers = [1, 30, 30, 30, 30, 30, 30, 1]#30, 30, 30, 30, \
                #  30, 30, 30, 30, 30, 30, 30, 30, 30, 30, \
                #  30, 30, 30, 30, 30, 30, 30, 30, 30, 30, \
                #  30, 30, 30, 30, 30, 30, 30, 30, 30, 30, \
                #  30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 
    # Set NN Structure
    
    data = scipy.io.loadmat('../data/PINC_mu3.mat')
    t_obtain = data['t'] # 
    X_obtain = data['x_10'] # 
    t_star = t_obtain[0:3000]
    # t_star = t_star 
    X_star = X_obtain[0:3000]
    # print(sol.t)
    # print(sol.y[0])
    N = X_star.shape[0]
    T = t_star.shape[0]
    # print(X_star)
    # print(t_star)
    # Rearrange Data 
    XX = np.tile(X_star, (1,T)) # [0:3000]
    TT = np.tile(t_star, (1,N)).T # 
    
    x = XX.flatten()[:,None] #
    t = TT.flatten()[:,None] #

    # Training Data    
    idx = np.random.choice(N*T, int(N*T*0.75), replace=False)
    x_train = X_star #x[:,:] # [idx,:]
    # noise = 0.00
    # x_train = x_train * (1 + noise*np.random.standard_normal(3000))
    t_train = t_star #t[:,:]

    # Training
    t_tic = time.time()
    model = DeepvdP(x_train, t_train, layers)
    model.train(200000)
    # cpu_time = timeit.default_timer()
    # Prediction
    x_pred = model.predict(t_star)
    # loss_hist = model.callback()
    elapsed_toc = time.time() - t_tic

    np.savetxt("prediction.txt", np.hstack(x_pred))
    print(elapsed_toc)
