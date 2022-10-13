# Explanation of the code and the idea behind it is found in the notebook.
# Also more etensive commenting is provided there.
# Running this file will train the network on 5000 generated samples, test on 500 and
# finally visualize the input along with the prediction for the last 5 generated test samples

from random import triangular
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

conv_vars = {
    'conv_padding': 0, 
    'conv_stride': 1,
    'conv_kernel': np.random.uniform(0, 1, (3,3)),
    'conv_bias': 1,
    'kernel_pool_size': 2,
    'kernel_pool_stride': 1,
    'weights_fc': np.random.uniform(0, 1, 4),  # weights for fully connected layer,
    'bias_fc': 0,
}

cross_indices = [
    [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2]
]
rectangle_indices = [
    [1, 0] , [1, 1], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0]
]

def generic_alt(arr):
    # make decision on one or two dots and create alteration in array
    num_dots = np.random.randint(1, 3)
    label=num_dots-1
    if label==0:
        # triangle
        for index in cross_indices:
            arr[index[0], index[1]] = np.random.uniform(0, 1)
    else:
        for index in rectangle_indices:
            arr[index[0], index[1]] = np.random.uniform(0, 1)
    
    return arr, label
           
def generate_input(size=(5,5)):
    img, label = generic_alt(np.zeros(size))
    return img, label


    
def visualize_img(arr, label, num=False):
    fig, axs = plt.subplots(1,1)
    axs.axis("off")
    axs.title.set_text(label)
    plt.imshow(arr)
    if num:
        for i in range(arr.shape[1]):
            for j in range(arr.shape[0]):
                axs.text(j,i,f"{arr[i,j]:.2f}",color="black",fontsize="small",ha="center",va="center")

    plt.show()
    
    
def convolve(X, k=conv_vars['conv_kernel'], p=conv_vars['conv_padding'], s=conv_vars['conv_stride'], b=0):
    # size of output matrix of convolution
    w_out = (X.shape[0]-k.shape[0]+2*p)//s + 1
    h_out = (X.shape[1]-k.shape[1]+2*p)//s + 1
    out = np.empty((w_out, h_out))
    
    for i in range(0, h_out, s):
        for j in range(0, w_out, s):
            out[i,j] = np.dot(X[i:i+k.shape[1], j:j+k.shape[0]].flatten(), k.flatten()) + b            
    return out

def relu(x):
    return np.max((0, x))

def d_relu(x):
    return np.heaviside(x, 0)  # assume value 0 for d'(0)

def max_pooling(X, k_size=conv_vars['kernel_pool_size'], s=conv_vars['kernel_pool_stride']):
    w_out = (X.shape[0]-k_size)//s + 1
    h_out = (X.shape[1]-k_size)//s + 1
    out = np.empty((w_out, h_out))

    indices = [0]*w_out*h_out # index matrix for backprop
    it = 0
    for i in range(0, h_out):
        for j in range(0, w_out):
            arr = X[i:i+k_size, j:j+k_size]
            out[i,j] = np.max(arr)
            indices[it] = np.add(np.unravel_index(arr.argmax(), arr.shape), (i, j))
            it+=1

    return out, indices


def bce(f_x, y):
    # binary cross entropy loss
    if 1-f_x < tol_loss: return 0
    return -(y*np.log(f_x) + (1-y)*np.log(1-f_x))

def d_bce(f_x, y):
    # derivative of bce loss function
    if 1-f_x < tol_loss: return 0
    return (-y/f_x + (1-y)/(1-f_x))

def sigmoid(x):
    ex=np.exp(x)
    return ex/(1+ex)

def d_sigmoid(x):
    sig=sigmoid(x)
    return sig*(1.0-sig)

def under_tol(w, b, tolerance):
    w[np.abs(w) < tolerance] = 0
    if b < tolerance: b = 0
    return w, b

train_it = 5000
test_it = 500
visual_it = 7  # how many examples to visualize when training and testing is completed
learn_rate = 0.05
tol_derivative = 1e-4
tol_loss = 1e-6
correct = 0
losses = np.empty(train_it)

for i in range(train_it+test_it+visual_it):
    
    input_img, y = generate_input()
    convolved_mat = convolve(input_img)

    # apply relu function
    relu_func = np.vectorize(relu)
    convolved_mat_relu = relu_func(convolved_mat)

    # perform pooling
    pooled_mat, pool_indices = max_pooling(convolved_mat_relu)


    # pass to "fully connected" - only one layer 
    input_fc = pooled_mat.flatten()
    weights_fc = conv_vars["weights_fc"]
    bias_fc = conv_vars["bias_fc"]

    # get activation with sigmoid function
    z = np.dot(input_fc, weights_fc) + bias_fc
    F = sigmoid(z)
    if i < train_it:
        # in training
        L = bce(F, y)
        losses[i] = L
        dL_dw = d_bce(F,  y)*d_sigmoid(z)*input_fc
        dL_dbf = d_bce(F, y)*d_sigmoid(z)*1
        dL_dw, dL_dbf = under_tol(dL_dw, dL_dbf, tol_derivative)
        #update weights

        conv_vars['weights_fc'] -= learn_rate*dL_dw
        conv_vars['bias_fc'] -= learn_rate*dL_dbf

        # Only indexes appearing as max in output of pooling affect output, and therefore
        dP =  np.zeros(convolved_mat_relu.shape) # derivative backpropagated through pooling layer
        for indices, dw in zip(pool_indices, dL_dw):
            i, j = indices[0], indices[1]
            dP[i, j] = dw



        # derivative of relu
        d_relu_func = np.vectorize(d_relu)
        dR = d_relu_func(convolved_mat) * dP



        # derivative of loss wrt. input
        dL_dF = convolve(input_img, k=dR)
        dL_db = np.sum(dR)
        dL_dF, dL_db = under_tol(dL_dF, dL_db, tol_derivative)

        conv_vars['conv_kernel'] -= learn_rate*dL_dF
        conv_vars['conv_bias'] -= learn_rate*dL_db
    else:
        # in testing
        p = round(F)
        if p == y:
            correct += 1
        if i > train_it + test_it:
            visualize_img(input_img, 'rectangle' if p==1 else 'cross')
print("Performance on test set: ", str(correct/test_it))





