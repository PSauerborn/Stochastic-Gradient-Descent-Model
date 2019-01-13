# Stochastic-Gradient-Descent-Model
An extension of previous models built that allows for stochastic gradient descent 

The following model uses Tensorflows low level API to construct an MLP (fully connected neural net) that can be used for multi-class 
classification. The model uses stochastic gradient descent and an Adam Optimization routine to minimize a softmax cross entropy
objective function, and also uses drop out as a regularization method.

The model was originally built to classify the outcome of crimes as part of a kaggle competition from a dataset that can be found at

https://www.kaggle.com/c/sf-crime/data

The competition used log loss as a metric, which appears in the model. However, with the exception of the log loss function (which can 
simply be removed without affecting any of the code) the model is completely general and can generalized for any classifcation task.

The model is built by passing down a dictionary containing the specifics of the desired architecture. The dictionary has the form

dict = {'n_layers': n, 'n_units': [a,b,...], 'activation_fn': [f1, f2,....]}

The model also comes with a batch generator to allow stochastic gradient descent; the batch size is passed down when the training method is 
called. See model doc strings for details

Note that the model also comes with a progress bar that updates as the model propogates through the mini-batches; this can also be removed 
easily without affecting any other code if desired


