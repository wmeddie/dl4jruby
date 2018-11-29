require 'jbundler'
require 'java'

java_import 'org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator'
java_import 'org.deeplearning4j.nn.conf.NeuralNetConfiguration'
java_import 'org.deeplearning4j.nn.conf.inputs.InputType'
java_import 'org.deeplearning4j.nn.conf.layers.DenseLayer'
java_import 'org.deeplearning4j.nn.conf.layers.OutputLayer'
java_import 'org.deeplearning4j.nn.conf.layers.ConvolutionLayer'
java_import 'org.deeplearning4j.nn.conf.layers.SubsamplingLayer'
java_import 'org.deeplearning4j.nn.conf.layers.PoolingType'
java_import 'org.deeplearning4j.nn.multilayer.MultiLayerNetwork'
java_import 'org.deeplearning4j.nn.weights.WeightInit'
java_import 'org.deeplearning4j.optimize.listeners.PerformanceListener'
java_import 'org.nd4j.linalg.activations.Activation'
java_import 'org.nd4j.linalg.learning.config.AMSGrad'
java_import 'org.nd4j.linalg.lossfunctions.LossFunctions'

batchSize = 256
trainData = MnistDataSetIterator.new(batchSize, true, 42)
nClasses = 10

conf = NeuralNetConfiguration::Builder.new()
    .updater(AMSGrad.new(0.005))
    .l2(5e-4)
    .weightInit(WeightInit::XAVIER)
    .activation(Activation::RELU).list(
      ConvolutionLayer::Builder.new(5, 5).nOut(20).build(),
      SubsamplingLayer::Builder.new(PoolingType::MAX).kernelSize(2, 2).build(),
      ConvolutionLayer::Builder.new(5, 5).nOut(50).build(),
      SubsamplingLayer::Builder.new(PoolingType::MAX).kernelSize(2, 2).build(),
      DenseLayer::Builder.new().nOut(500).activation(Activation::RELU).build(),
      OutputLayer::Builder.new(LossFunctions::LossFunction::NEGATIVELOGLIKELIHOOD).nOut(nClasses).activation(Activation::SOFTMAX).build()
    )
    .setInputType(InputType.convolutionalFlat(28, 28, 1))
    .build()

model = MultiLayerNetwork.new(conf)
model.setListeners(PerformanceListener.new(10, true))
model.init()
puts model.summary

model.fit(trainData, 3)

testData = MnistDataSetIterator.new(batchSize, false, 42)
eval = model.evaluate(testData)
puts eval.toString
