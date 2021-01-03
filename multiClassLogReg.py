import numpy as np
#import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets
import sklearn.model_selection
#import timeit


# def sigmoid(u): #signmoid function
#     expu = np.exp(u)
#     return expu / (1 + expu)

def softmax(u):
    
    expu = np.exp(u)
    denom = np.sum(expu)
    q = expu/denom
    return q

# def softmax_slow(u):
#     K = len(u)
#     result = np.zeros(K)
#     denom = 0
    
#     for k in range(K):
#         denom = denom + np.exp(u[k])
        
#     for k in range(K):
#         result[k] = np.exp(u[k])/denom
#     return result

# u = np.random.randn(1000000)
# t1 = timeit.default_timer()
# result = softmax(u)
# t2 = timeit.default_timer()
# print("time: ", t2 - t1)

# u = np.random.randn(1000000)
# t1 = timeit.default_timer()
# result = softmax_slow(u)
# t2 = timeit.default_timer()
# print("time: ", t2 - t1)

#function for multi class classification 
# def cross_entropy_slow(p,q): #result of softmax function is q
#     K = len(p)
#     result = 0
    
#     for k in range(K):
#         result = result - p[k]*np.log(q[k])
    # return result 

# u1 = np.random.randn(5)
# p = softmax(u1) #probability vector --- adds up to 1
# u2 = np.random.randn(5)
# q = softmax(u2) 
# check = cross_entropy(p, q)
# print("check is: ", check)



#write non for loop version -- try that with random sample

def cross_entropy(p,q):
    
   result = np.vdot(-p, np.log(q))
   return result
   
u1 = np.random.randn(5)
p = softmax(u1) #probability vector --- adds up to 1
u2 = np.random.randn(5)
q = softmax(u2) 


def f(x, beta): #prediciton
    
    K = beta.shape[0]
    u = np.zeros(K) # before softmax
    d = len(x)
    
    for k in range(K):
        u[k] = beta[k,0]
        for j in range(d): # j+1 bc it starts at 0
            u[k] += beta[k,j+1] * x[j]
            
    return softmax(u)
    
  
    
def eval_L(beta, X, Y): #average cross entropy loss
    
    N, d = X.shape
    s = 0 #sum
    
    for i in range(N):
        Xi = X[i]
        Yi = Y[i]
        s = s + cross_entropy(Yi, f(Xi, beta))
    return s/N

dataset = sk.datasets.load_iris()
X = dataset.data
Y = dataset.target
K = 3
d = 4
N = X.shape[0]
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X,Y)
N_train = X_train.shape[0]
N_test = X_test.shape[0]
print("Hello World")
#%%
Y_one_hot = np.zeros((N_train, K))
for i in range(N_train):
    k = Y_train[i]
    Y_one_hot[i, k] = 1
np.random.seed(123)
#%%
#X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X,Y_one_hot)
beta = np.zeros((K,d+1))
L = eval_L(beta, X_train, Y_one_hot)
L_vals = [L]

#beta values
for i in range(100):
    delta_beta = np.random.randn(K, d+1)
    candidate_beta = beta + (0.05*delta_beta)
    candidate_L = eval_L(candidate_beta, X_train, Y_one_hot)
    if candidate_L < L:
        beta = candidate_beta
        L = candidate_L
    L_vals.append(L)
    print(L)


Y_pred = np.zeros(N_test)

for i in range(N_test):
    Xi = X_test[i]
    f_val = f(Xi, beta) #3 components , probability vector 
    Y_pred[i] = np.argmax(f_val)
    
num_correct = np.sum(Y_pred == Y_test)
acc = num_correct / N_test
print(acc)   








    