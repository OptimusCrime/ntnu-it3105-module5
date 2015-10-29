__author__ = 'keithd'

import os
import theano
import theano.tensor as T
import theano.tensor.nnet as Tann
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor.nnet as Tann
#import grapher as graph
import graphviz # Needed for the import of pydot
import pydot # for printing out theano function graphs to a file

debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug/')

# Theano pretty-print
def ppth(obj, fancy=True, graph=False, fid=debug_dir,fmt='pdf'):
    if graph: theano.printing.pydotprint(obj,outfile=fid + 'test123',format=fmt)
    elif fancy: theano.printing.debugprint(obj)
    else: return theano.pp(obj)

 # ***** EXAMPLES **********
# A series of theano examples used in my theano lecture for AI Programming.

def theg1():
  w = T.bscalar('w')
  x = T.bscalar('x')
  y = T.bscalar('y')
  z = w*x + y
  f = theano.function([w,x,y],z)
  ppth(z,graph=True)
  return f

def theg2():
  w = T.dscalar('w')
  x = T.dscalar('x')
  y = T.dscalar('y')
  z = 7*w*x + y
  f = theano.function([w,x,y],z)
  dz = T.grad(z,x)
  g = theano.function([w,x,y],dz)
  ppth(dz,graph=True)
  return g

# Now our variables represent matrices
def theg3():
    w = T.dmatrix('weights')  # matrix of 64-bit (double) floats
    v = T.dvector('upstream activations')  # vector of "     "          "
    b = T.dvector('biases')
    x = T.dot(v,w) + b  #T.dot = tensor dot product
    x.name = 'integrated signals'
    f = theano.function([v,w,b],x)
    ppth(x,graph=True)
    return f

# Matrix operations with a state variable, s.
def theg4(n=10):
    w = T.dmatrix('weights')  # matrix of 64-bit (double) floats
    v = T.dvector('upstream activations')  # vector of "     "          "
    b = T.dvector('biases')
    s = theano.shared(np.zeros(2)) # s = state = accumulator of x
    x = T.dot(v,w) + b  #T.dot = tensor dot product
    x.name = 'integrated signals'
    f = theano.function([v,w,b],x,updates=[(s,s+x)])
    w0 = np.random.uniform(-.1,.1,size=(2,2)) # init wgt vector
    b0 = [1,1]  # init bias
    ppth(f,graph=True)
    for i in range(n): f([1+i/n,1-i/n],w0,b0)
    return (f,s)

# This just builds all the functionality around a perceptron, but it never executes
def theg5(target=[1,1]):
    w = theano.shared(np.random.uniform(-.1,.1, size=(2,2)))
    v = T.dvector('V') ; b = theano.shared(np.ones(2))
    x = Tann.sigmoid(T.dot(v,w) + b)
    w.name = 'w'; x.name = 'x'
    error = T.sum((target-x)**2)
    de = T.grad(error,w)
    ppth(error,graph=True)
    return (x,de)

def gen_all_bit_cases(num_bits):
    def bits(n):
        s = bin(n)[2:]
        return [int(b) for b in '0'*(num_bits - len(s))+s]
    return [bits(i) for i in range(2**num_bits)]

def vector_distance(vect1,vect2):
    return (sum([(v1 - v2)**2 for v1,v2 in zip(vect1,vect2)]))**0.5

def calc_avg_vect_dist(vectors):
    n = len(vectors); sum = 0
    for i in range(n):
        for j in range(i+1,n):
            sum += vector_distance(vectors[i],vectors[j])
    return 2*sum/(n*(n-1))


class autoencoder():

    # nb = # bits, nh = # hidden nodes (in the single hidden layer)
    # lr = learning rate

    def __init__(self,nb=3,nh=2,lr=.1):
        self.cases = gen_all_bit_cases(nb)
        print self.cases
        self.lrate = lr
        self.build_ann(nb,nh,lr)

    def build_ann(self,nb,nh,lr):
        w1 = theano.shared(np.random.uniform(-.1,.1,size=(nb,nh)))
        w2 = theano.shared(np.random.uniform(-.1,.1,size=(nh,nb)))
        input = T.dvector('input')
        b1 = theano.shared(np.random.uniform(-.1,.1,size=nh))
        b2 = theano.shared(np.random.uniform(-.1,.1,size=nb))
        x1 = Tann.sigmoid(T.dot(input,w1) + b1)
        x2 = Tann.sigmoid(T.dot(x1,w2) + b2)
        error = T.sum((input - x2)**2)
        params = [w1,b1,w2,b2]
        gradients = T.grad(error,params)
        backprop_acts = [(p, p - self.lrate*g) for p,g in zip(params,gradients)]
        self.predictor = theano.function([input],[x2,x1])
        self.trainer = theano.function([input],error,updates=backprop_acts)

        ppth(self.trainer,graph=True)

    def do_training(self,epochs=100,test_interval=None):
        errors = []
        for i in range(epochs):
            error = 0
            for c in self.cases:
                print 'case = '
                print c
                error += self.trainer(c)
                print error
                print '----'
            errors.append(error)

        print 'errors = '
        print errors

    def do_testing(self,scatter=True):
        hidden_activations = []
        for c in self.cases:
            _,hact = self.predictor(c)
            hidden_activations.append(hact)
        if scatter: graph.simple_scatter(hidden_activations,radius=8)
        return hidden_activations

def autotest(nb=3,nh=2,lr=.1,epochs=100,ti=10):
    ac = autoencoder(nb,nh,lr)
    graph.newfig()
    ac.do_training(epochs,test_interval=ti)
    graph.newfig()
    return ac.do_testing()
