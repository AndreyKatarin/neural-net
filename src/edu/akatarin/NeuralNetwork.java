package edu.akatarin;

import java.util.Arrays;

public class NeuralNetwork {
    private final static double LEARNING_RATE = 0.01;
    private final static double MOMENTUM = 0.7;
    private final static int INPUT_LAYER_SIZE = 784;
    private final static int HIDDEN_LAYER_SIZE = 800;
    private final static int OUTPUT_LAYER_SIZE = 10;

    private final static double[][] ideals = {
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},//0
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},//1
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},//2
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},//3
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},//4
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},//5
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},//6
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},//7
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},//8
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},//9
    };

    private final NeuronLayer inputLayer;
    private final NeuronLayer hiddenLayer;
    private final NeuronLayer outputLayer;

    public NeuralNetwork() {
        this.inputLayer = new NeuronLayer(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
        this.hiddenLayer = new NeuronLayer(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
        this.outputLayer = new NeuronLayer(OUTPUT_LAYER_SIZE, 0);
    }

    public void feedForward(Number number) {
        double[] pixels = number.getPixels();
        inputLayer.setNeurons(pixels);
        hiddenLayer.setNeurons(inputLayer.calculateOutput());
        outputLayer.setNeurons(hiddenLayer.calculateOutput());
    }

    public double[] getOutput() {
        return Arrays.stream(outputLayer.getNeurons())
                .mapToDouble(Neuron::getValue).toArray();
    }

    public double getMSE(Number number) {
        double[] idealOut = ideals[number.getValue()];
        double[] currentOutput = getOutput();
        double total = 0;
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            total += Math.pow((idealOut[i] - currentOutput[i]), 2);
        }
        return total / OUTPUT_LAYER_SIZE;
    }

    //производная от функции активации(сигмоида)
    private double sigmDx(double x) {
        return x * (1 - x);
    }

    //дельты(ошибки) значений нейронов выходного слоя
    private double[] calcOutputWeightsDelta(Number number) {
        double[] deltas = new double[OUTPUT_LAYER_SIZE];
        double[] idealOut = ideals[number.getValue()];
        double[] outputs = getOutput();
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            double error = idealOut[i] - outputs[i];
            deltas[i] = error * sigmDx(outputs[i]);
        }
        return deltas;
    }

    private double[] calcHiddenWeightsL1Delta(double[] outputsDelta) {
        double[] deltas = new double[HIDDEN_LAYER_SIZE];
        Neuron[] output = hiddenLayer.getNeurons();
        double[][] weights = hiddenLayer.getWeights();//800x10
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            double error = 0;
            double[] hiddenNeuronWeights = weights[i];//10
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                error += outputsDelta[j] * hiddenNeuronWeights[j];
            }
            deltas[i] = error * sigmDx(output[i].getValue());
        }
        return deltas;
    }

    /*
    изменение веса синапса равно коэффициенту скорости обучения, умноженному на градиент этого веса,
    прибавить момент умноженный на предыдущее изменение этого веса (на 1-ой итерации равно 0)
     */
    private void updateInputLayerWeights(double[] hiddenL1WeightsDelta) {
        double[][] deltas = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
        double[] biasDelta = new double[HIDDEN_LAYER_SIZE];
        Neuron[] neurons = inputLayer.getNeurons();
        double[][] prevWeightDeltas = inputLayer.getWeightDeltas();
        double[] prevBiasDeltas = inputLayer.getBiasDeltas();
        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                double gradient = neurons[i].getValue() * hiddenL1WeightsDelta[j];
                deltas[i][j] = (gradient * LEARNING_RATE) + (MOMENTUM * prevWeightDeltas[i][j]);
                biasDelta[j] = (hiddenL1WeightsDelta[j] * LEARNING_RATE) + (MOMENTUM * prevBiasDeltas[j]);
            }
        }
        inputLayer.updateWeights(deltas);
        inputLayer.updateBias(biasDelta);
    }

    private void updateHiddenLayerWeights(double[] outputsDelta) {
        double[][] deltas = new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];
        double[] biasDelta = new double[OUTPUT_LAYER_SIZE];
        Neuron[] neurons = hiddenLayer.getNeurons();
        double[][] prevWeightDeltas = hiddenLayer.getWeightDeltas();
        double[] prevBiasDeltas = hiddenLayer.getBiasDeltas();
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                double gradient = neurons[i].getValue() * outputsDelta[j];
                deltas[i][j] = (gradient * LEARNING_RATE) + (MOMENTUM * prevWeightDeltas[i][j]);
                biasDelta[j] = (outputsDelta[j] * LEARNING_RATE) + (MOMENTUM * prevBiasDeltas[j]);
            }
        }
        hiddenLayer.updateWeights(deltas);
        hiddenLayer.updateBias(biasDelta);
    }

    public void backpropagation(Number number) {
        double[] outputsDelta = calcOutputWeightsDelta(number);
        double[] hiddenDelta = calcHiddenWeightsL1Delta(outputsDelta);

        updateHiddenLayerWeights(outputsDelta);
        updateInputLayerWeights(hiddenDelta);

    }
}
