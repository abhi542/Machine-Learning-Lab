import numpy as np

X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)
X /= np.amax(X, axis=0)
y /= 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

epochs = 5000
lr = 0.1
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

wh = np.random.uniform(size=(input_neurons, hidden_neurons))    # I, H
bh = np.random.uniform(size=(1, hidden_neurons))                # 1, H
wout = np.random.uniform(size=(hidden_neurons, output_neurons)) # H, O
bout = np.random.uniform(size=(1, output_neurons))              # 1, O

for i in range(epochs):
    
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hl = sigmoid(hinp)
    outinp1 = np.dot(hl, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)
    
    EO = y - output
    outgrad = sigmoid_derivative(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    
    hiddengrad = sigmoid_derivative(hl)
    d_hidden = EH * hiddengrad
    
    wout += hl.T.dot(d_output) * lr
    wh += X.T.dot(d_hidden) * lr

print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)