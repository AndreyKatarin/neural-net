package edu.akatarin;

import java.util.Arrays;

public class NeuralNetwork {
    private final static double LEARNING_RATE = 0.01;
    private final static double MOMENTUM = 0;
    private final static int INPUT_LAYER_SIZE = 784;
    private final static int HIDDEN_LAYER_SIZE = 800;
    private final static int OUTPUT_LAYER_SIZE = 10;

    private final static double[][] ideals = {
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
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
        return (1 - x) * x;
    }

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

    private double[] calcHiddenWeightsDelta(double[] outputsDelta) {
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

//    private double[][] calcHiddenLayerWeightGradients(double[] outputsDelta) {
//        double[][] gradients = new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];
//        Neuron[] output = hiddenLayer.getNeurons();
//        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
//            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
//                gradients[i][j] = output[i].getValue() * outputsDelta[j];
//            }
//        }
//        return gradients;
//    }
//
//    private double[][] calcHiddenLayerWeightDeltas(double[][] hiddenLayerWeightGradients) {
//        double[][] deltas = new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];
//        double[][] prevDeltas = hiddenLayer.getWeightDeltas();
//        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
//            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
//                deltas[i][j] = LEARNING_RATE * hiddenLayerWeightGradients[i][j] + MOMENTUM * prevDeltas[i][j];
//            }
//        }
//        return deltas;
//    }
//
//    private double[][] calcInputLayerWeightGradients(double[] hiddenDelta) {
//        double[][] gradients = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
//        Neuron[] output = inputLayer.getNeurons();
//        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
//            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
//                gradients[i][j] = output[i].getValue() * hiddenDelta[j];
//            }
//        }
//        return gradients;
//    }
//
//    private double[][] calcInputLayerWeightDeltas(double[][] inputLayerWeightGradients) {
//        double[][] deltas = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
//        double[][] prevDeltas = inputLayer.getWeightDeltas();
//        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
//            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
//                deltas[i][j] = LEARNING_RATE * inputLayerWeightGradients[i][j] + MOMENTUM * prevDeltas[i][j];
//            }
//        }
//        return deltas;
//    }

    private void updateInputLayerWeights(double[] hiddenWeightsDelta){
        double[][] deltas = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
        Neuron[] neurons = inputLayer.getNeurons();
        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                deltas[i][j] = neurons[i].getValue()*hiddenWeightsDelta[j]*LEARNING_RATE;
            }
        }
        inputLayer.updateWeights(deltas);
    }

    private void updateHiddenLayerWeights(double[] outputWeightsDelta){
        double[][] deltas = new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];
        Neuron[] neurons = hiddenLayer.getNeurons();
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                deltas[i][j] = neurons[i].getValue()*outputWeightsDelta[j]*LEARNING_RATE;
            }
        }
        hiddenLayer.updateWeights(deltas);
    }

    public void backpropagation(Number number) {
        double[] outputsDelta = calcOutputWeightsDelta(number);
        updateHiddenLayerWeights(outputsDelta);
        double[] hiddenDelta = calcHiddenWeightsDelta(outputsDelta);
        updateInputLayerWeights(hiddenDelta);

//        double[][] hiddenLayerWeightGradients = calcHiddenLayerWeightGradients(outputsDelta);
//        double[][] hiddenLayerWeightDeltas = calcHiddenLayerWeightDeltas(hiddenLayerWeightGradients);
//        hiddenLayer.updateWeights(hiddenLayerWeightDeltas);
//
//        double[][] inputLayerWeightGradients = calcInputLayerWeightGradients(hiddenDelta);
//        double[][] inputLayerWeightDeltas = calcInputLayerWeightDeltas(inputLayerWeightGradients);
//        inputLayer.updateWeights(inputLayerWeightDeltas);
    }
}
