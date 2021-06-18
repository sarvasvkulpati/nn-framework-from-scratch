# Neural Network From Scratch

I'm building this library to reimplement machine learning concepts I know *from scratch*, so that I can be sure I understand how they work.

Currently, everything is located in a single nn.py file. I'll probably structure it better as I add to it.



Included at the moment are:
- Dense layers
- Softmax and ReLU activations
- Dropout
- Categorical Crossentropy loss
- SGD
- Adagrad
- RMSProp
- Momentum
- Adam

Example: 
```python

# define layers
dense1 = Layer_Dense(2, 512, wl2=5e-4, bl2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10001):

    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)

    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    loss = data_loss + regularization_loss


    predictions = np.argmax(loss_activation.output, axis=1)

    # check progress
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}'  + f'loss: {loss:.3f}' + f'lr: {optimizer.current_learning_rate:.3f}')


    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()



```
