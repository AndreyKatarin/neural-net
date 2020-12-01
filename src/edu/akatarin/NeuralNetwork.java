package edu.akatarin;


import java.util.Arrays;

public class NeuralNetwork {
    private CostFunction costFunction = CostFunction.MSE;//default
    private Optimizer optimizer = new Optimizer.GradientDescent(0.5);
    private double momentum = 0.5;

    private final NeuronLayer inputLayer;
    private final NeuronLayer[] hiddenLayers;
    private final NeuronLayer outputLayer;

    public NeuralNetwork(NeuronLayer inputLayer, NeuronLayer[] hiddenLayers, NeuronLayer outputLayer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
    }

    public void setCostFunction(CostFunction costFunction) {
        this.costFunction = costFunction;
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    //Batch Gradient Descent
    //обновляем веса входного и скрытого слоев на сумму DeltaW всех весов в пакете.
    public double trainBatch(Number[] numbers, double[][] expectedOut) {
        double totalBatchError = 0;
        for (int n = 0; n < numbers.length; n++) {
            Number number = numbers[n];
            feedForward(number);
            totalBatchError += getCost(expectedOut[n]);
            double[] expected = expectedOut[n];
            calcOutputLayerError(expected); //ошибка выходного слоя
            //распространяем ошибку на скрытые слои
            NeuronLayer lastHiddenLayer = hiddenLayers[hiddenLayers.length - 1];
            calcHiddenLayerError(lastHiddenLayer, outputLayer);
            if (hiddenLayers.length > 1) {
                int hiddenLayersToPropagate = hiddenLayers.length - 2;
                for (int i = hiddenLayersToPropagate; i >= 0; i--) {
                    NeuronLayer currentHiddenLayer = hiddenLayers[i];
                    NeuronLayer previousHiddenLayer = hiddenLayers[i + 1];
                    calcHiddenLayerError(currentHiddenLayer, previousHiddenLayer);
                }
            }

            //подсчитаем градиент изменения весов скрытого слоя
            NeuronLayer firstHiddenLayer = hiddenLayers[0];
            calcWeightsAndBiasesGradient(inputLayer, firstHiddenLayer);
            if (hiddenLayers.length > 1) {
                int hiddenLayersToUpdate = hiddenLayers.length - 2;
                for (int i = 0; i <= hiddenLayersToUpdate; i++) {
                    NeuronLayer currentHiddenLayer = hiddenLayers[i];
                    NeuronLayer nextHiddenLayer = hiddenLayers[i + 1];
                    calcWeightsAndBiasesGradient(currentHiddenLayer, nextHiddenLayer);
                }
            }
            calcWeightsAndBiasesGradient(lastHiddenLayer, outputLayer);
        }
        //обновим веса
        updateAllWeightsAndBiases();
        return totalBatchError;
    }

    public void feedForward(Number number) {
        feedForward(number.getPixels());
    }

    public void feedForward(double[] pixels) {
        inputLayer.setNeurons(pixels);
        NeuronLayer firstHiddenLayer = hiddenLayers[0];
        firstHiddenLayer.setNeurons(firstHiddenLayer.getActivation().apply(inputLayer.calculateNetOutput()));
        if (hiddenLayers.length > 1) {
            int hiddenLayersToFeed = hiddenLayers.length - 2;
            for (int i = 0; i <= hiddenLayersToFeed; i++) {
                NeuronLayer currentHiddenLayer = hiddenLayers[i];
                NeuronLayer nextHiddenLayer = hiddenLayers[i + 1];
                nextHiddenLayer.setNeurons(nextHiddenLayer.getActivation().apply(currentHiddenLayer.calculateNetOutput()));
            }
        }
        NeuronLayer lastHiddenLayer = hiddenLayers[hiddenLayers.length - 1];
        double[] res = lastHiddenLayer.calculateNetOutput();
        outputLayer.setNeurons(outputLayer.getActivation().apply(res));
    }

    public double[] getOutput() {
        return outputLayer.getOutput();
    }

    public double getCost(double[] idealOut) {
        double[] currentOutput = getOutput();
        return costFunction.apply(idealOut, currentOutput);
    }

    //ошибка значений нейронов выходного слоя
    private void calcOutputLayerError(double[] idealOut) {
        //How much does the cost change when the input to the last layer changes
        double[] outputs = outputLayer.getOutput();
        //How much does the cost change when the output from the neuron changes?
        double[] dCdO = costFunction.applyDerivative(idealOut, outputs);
        //How much does the output from the neuron change when the input changes?
        double[] dCdI = outputLayer.getActivation().applyDerivative(outputs, dCdO);
        outputLayer.setDeltas(dCdI);
    }

    //распространяем ошибку выходного слоя на скрытые слои (hid-n, out)
    private void calcHiddenLayerError(NeuronLayer currentLayer, NeuronLayer previousLayer) {
        int currentLayerSize = currentLayer.getSize();
        //dIHdWH - How much does the input value to the neuron change when wH changes?
        double[] output = Arrays.stream(currentLayer.getNeurons())
                .mapToDouble(Neuron::getValue)
                .toArray();
        double[][] weights = currentLayer.getWeights();
        //How much does the cost change when the input changes?
        double[] dCdI_prev = previousLayer.getDeltas();
        double[] dCdZ = new double[currentLayerSize];
        for (int i = 0; i < currentLayerSize; i++) {
            double[] dIdO = weights[i]; //веса ассоциированные с i-м нейроном
            double sum = 0;
            for (int j = 0; j < dCdI_prev.length; j++) {
                sum += dCdI_prev[j] * dIdO[j];
            }
            dCdZ[i] = sum;
        }
        //How much does the output from the neuron change when the input changes?
        double[] dCdI = currentLayer.getActivation().applyDerivative(output, dCdZ);
        currentLayer.setDeltas(dCdI);
    }

    //    изменение веса синапса равно коэффициенту скорости обучения, умноженному на градиент этого веса,
    //    прибавить момент умноженный на предыдущее изменение этого веса (на 1-ой итерации равно 0)
    private void calcWeightsAndBiasesGradient(NeuronLayer currentLayer, NeuronLayer nextLayer) {
        int size = currentLayer.getSize();
        int nextSize = currentLayer.getNextSize();
        double[][] currentWeightsDelta = new double[size][nextSize];
        double[][] previousWeightsDelta = currentLayer.getPrevWeightDeltas();
        double[] currentBiasDelta = new double[nextSize];
        double[] previousBiasDelta = currentLayer.getPrevBiasDeltas();
        Neuron[] neurons = currentLayer.getNeurons();
        double[] nextLayerError = nextLayer.getDeltas();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                //dCdW how much does the total cost change when exactly that W changes.
                double gradient = neurons[i].getValue() * nextLayerError[j];
                currentWeightsDelta[i][j] = optimizer.apply(gradient) + (momentum * previousWeightsDelta[i][j]);
            }
        }
        for (int i = 0; i < nextSize; i++) {
            currentBiasDelta[i] = optimizer.apply(nextLayerError[i]) + (momentum * previousBiasDelta[i]);
        }
        currentLayer.saveWeightDeltas(currentWeightsDelta);
        currentLayer.saveBiasDeltas(currentBiasDelta);
    }

    //Stochastic Gradient Descent
    //Веса обновляются в real-time, сразу после того как посчитано DeltaW
    public void backpropagation(double[] idealOut) {
        calcOutputLayerError(idealOut); //ошибка выходного слоя
        //распространяем ошибку на скрытые слои
        NeuronLayer lastHiddenLayer = hiddenLayers[hiddenLayers.length - 1];
        calcHiddenLayerError(lastHiddenLayer, outputLayer);
        if (hiddenLayers.length > 1) {
            int hiddenLayersToPropagate = hiddenLayers.length - 2;
            for (int i = hiddenLayersToPropagate; i >= 0; i--) {
                NeuronLayer currentHiddenLayer = hiddenLayers[i];
                NeuronLayer previousHiddenLayer = hiddenLayers[i + 1];
                calcHiddenLayerError(currentHiddenLayer, previousHiddenLayer);
            }
        }

        //подсчитаем градиент изменения весов скрытого слоя и сразу же обновим веса
        NeuronLayer firstHiddenLayer = hiddenLayers[0];
        calcWeightsAndBiasesGradient(inputLayer, firstHiddenLayer);
        if (hiddenLayers.length > 1) {
            int hiddenLayersToUpdate = hiddenLayers.length - 2;
            for (int i = 0; i <= hiddenLayersToUpdate; i++) {
                NeuronLayer currentHiddenLayer = hiddenLayers[i];
                NeuronLayer nextHiddenLayer = hiddenLayers[i + 1];
                calcWeightsAndBiasesGradient(currentHiddenLayer, nextHiddenLayer);
            }
        }
        calcWeightsAndBiasesGradient(lastHiddenLayer, outputLayer);
        updateAllWeightsAndBiases();
    }

    private void updateAllWeightsAndBiases() {
        inputLayer.update();
        if (hiddenLayers.length > 1) {
            int hiddenLayersToUpdate = hiddenLayers.length - 2;
            for (int i = 0; i <= hiddenLayersToUpdate; i++) {
                NeuronLayer currentHiddenLayer = hiddenLayers[i];
                currentHiddenLayer.update();
            }
        }
        NeuronLayer lastHiddenLayer = hiddenLayers[hiddenLayers.length - 1];
        lastHiddenLayer.update();
    }
}
