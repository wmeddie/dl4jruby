# dl4jruby

Ruby Wrapper for DL4J.  It will eventually provide an ergonomic interface to DL4J while still providing the same performance.

# Installing

DL4J uses maven for dependency management and this library uses jbundler to manage DL4J.

Before running any test scrips first install the jbundle with:

```
    $ jruby -S jbundle install
```

# Running

Once you have installed the jbundle, you can run the train_mnist.rb script with just `jruby train_mnist.rb`.  It will train a simple neural network on MNIST and get an accuracy of 98.7%.

# Going faster

You can speed up train_mnist by installing CUDA and CUDNN, and uncommenting out the commented out lines in the Jarfile.

