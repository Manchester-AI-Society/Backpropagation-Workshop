#set document(
  title: [Backpropagation workshop]
)

#title()

= Generalised backpropagation from mathematical principles

== Neural network forward pass

First, consider a neural network with $L$ layers.
Therefore each layer is defined as: 


$$$
text("The pre-activation function:") z^(l) = W^l x^(l-1) + b^l 
\
text("The post-activation output:") x^(l) = sigma ^l (z^l)
\
forall l >= L, l in NN
$$$
Where:

- $W^l$ are the *weights* for the specific layer 

- $b^l$ are the *biases* for the specific layer 

- $x^l$ are the *inputs* for the specific layer 

- $sigma ^l (z^l)$ is the *activation function* for the specific layer

=== The loss function

The *loss function* will be written as:

$$$
cal(L)(x^l , y)
$$$

The goal of backpropagation is to find *the gradients of the weights and biases* in the layers.
Backpropagation use *partial derivatives*, treating each weight as its own dimension within gradient descent, which was covered in the Gradient descent workshop.

The *gradients* to be worked out:
$$$
(partial cal(L))/(partial W^l) , (partial cal(L))/(partial b^l)
$$$

== Core backpropagation algorithm

=== The chain rule

As a precursor to this, we will define the chain rule, which backpropagation is essential an iterative process of:

$$$
(d cal(L))/(d x ) = (d cal(L))/(d y ) (d y)/(d x )
$$$

=== Output layer

As the output layer is the last layer, everything is derived from this as the error function can only deal with the errors from the last layer:

$$$
delta ^L = (partial cal(L))/(partial z^L) 
\
==>  delta ^L = (partial cal(L))/(partial a^L) dot.o sigma '^((L))(z^L)
$$$

=== Hidden layers

The hidden layer gradients are worked out recursively through this equation.

For layer $l$, while $l<L$:
$$$
delta ^l = (W^(l+1))^T delta ^(l+1) dot.o sigma '^((l))(z^l)
$$$

This propagates the error backwards and multiplies by the derivative of the activation.

This also highlights why long term dependencies fail at RNN as if the eigenvalue of the weights is above or below 1, the neural network will explode or minimise.

== Gradient descent 

$delta^l$ is now calculated, therefore the weight and bias gradients are:

$$$
(partial cal(L))/(partial W^l)  = delta^l dot (a^(l-1))^T
\
(partial cal(L))/(partial b^l)  = delta^l 
$$$

= Practical mathematical example of backpropagation

Let us define a 2 layer sigmoid function neural network.

The neural network will have:
- Input features: 2 neurons
- Hidden layer: 2 neurons
- Output layer: 1 neuron
- Activation function: $sigma(x) = 1/(1+e^(-x))$
- Loss: Mean squared error ($cal(L) = 1/2 (y-hat(y))^2$)


First we will define all the mathematical notation we need.

$$$
bb(x) = [x_1,x_2]^T
\
W^1 = mat(w_11,w_12 ; w_21,w_22)
\
b^1 = mat(b_1,b_2 )^T
\
z^1 = sigma(W^1 bb(x) + b^1)
$$$


== Forward pass example

Let the network have input $bold(x) = [x_1, x_2]^T$, hidden layer weights $W_1$, biases $b_1$, output weights $W_2$ and bias $b_2$:

$ bold(h) = sigma(W_1 bold(x) + b_1), quad hat(y) = sigma(W_2 bold(h) + b_2) $

Take a simple example:

$ bold(x) = [0.05, 0.10], quad y = 1 $

$ W_1 = mat(0.15, 0.20; 0.25, 0.30), quad b_1 = vec(0.35, 0.35), quad W_2 = [0.40, 0.45], quad b_2 = 0.60 $

Compute hidden pre-activations:

$ z_1 = W_1 bold(x) + b_1 = vec(0.3775, 0.3925), quad bold(h) = sigma(z_1) approx vec(0.59327, 0.59688) $

Compute output:

$ z_2 = W_2 bold(h) + b_2 approx 1.10591, quad hat(y) = sigma(z_2) approx 0.75137 $

Loss (MSE):

$ cal(L) = 1/2 (y - hat(y))^2 approx 0.0318 $

== Output layer

As the output layer is the last layer, the error is computed from the loss:

$ delta^L = (partial cal(L)) / (partial z^L) quad arrow.double quad delta^L = (partial cal(L)) / (partial a^L) dot.o sigma'(z^L) $

For our example:

$ delta_2 = (hat(y) - y) dot hat(y)(1 - hat(y)) approx -0.0461 $

Gradients for output weights and bias:

$ (partial cal(L)) / (partial W_2) = delta_2 dot bold(h)^T approx [-0.0273, -0.0275], quad (partial cal(L)) / (partial b_2) = delta_2 approx -0.0461 $

== Hidden layers

The hidden layer gradients are worked out recursively:

$ delta^l = (W^(l+1))^T delta^(l+1) dot.o sigma'(z^l) $

For our hidden layer:

$ sigma'(z_1) = bold(h) dot.o (1 - bold(h)) approx [0.2413, 0.2406] $

$ delta_1 = vec(-0.00445, -0.00499), quad (partial cal(L)) / (partial W_1) = delta_1 dot bold(x)^T approx mat(-0.000223, -0.000445; -0.000249, -0.000499), quad (partial cal(L)) / (partial b_1) = delta_1 $


== Gradient descent updating the weights using the gradients

With learning rate $eta = 0.5$:

$ W_2 = W_2 - eta (partial cal(L)) / (partial W_2) approx [0.41365, 0.46375], quad b_2 = 0.62305 $

$ W_1 = W_1 - eta (partial cal(L)) / (partial W_1) approx mat(0.15011, 0.19978; 0.24987, 0.29975), quad b_1 =  vec(0.3522, 0.3525) $


