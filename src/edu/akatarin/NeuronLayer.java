package edu.akatarin;

import java.util.Random;

public class NeuronLayer {

    private final int size;
    private final int nextSize;
    private final Neuron[] neurons;
    private final double[][] weights;
    private double[][] weightDeltas;

    public NeuronLayer(int size, int nextSize) {
        Random r = new Random();
        this.size = size;
        this.nextSize = nextSize;
        weights = new double[size][nextSize];
        weightDeltas = new double[size][nextSize];
        neurons = new Neuron[size];
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron();
            for (int j = 0; j < nextSize; j++) {
                weights[i][j] = r.nextGaussian();
            }
        }
    }

    public double[][] getWeightDeltas() {
        return weightDeltas;
    }

    public void updateWeights (double[][] weightDeltas) {
        this.weightDeltas = weightDeltas;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights[i][j] = weights[i][j] + weightDeltas[i][j];
            }
        }
    }

    public double[][] getWeights() {
        return weights;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public void setNeurons(double[] inputs) {
        for (int i = 0; i < size; i++) {
            neurons[i].setValue(inputs[i]);
        }
    }

    public double[] calculateOutput() {
        double[] output = new double[nextSize];
        double total = 0.0;
        for (int i = 0; i < nextSize; i++) {
            for (int j = 0; j < size; j++) {
                total += neurons[j].getValue() * weights[j][i];
            }
            output[i] = activation(total);
        }
        return output;
    }

    private double activation(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
