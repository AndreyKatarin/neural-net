package edu.akatarin;


import java.util.Arrays;

public class NeuronLayer {

    private Activation activation = Activation.Sigmoid;//default
    private Initializer initializer = Initializer.RANDOM_GAUSSIAN;
    private final int size;
    private final int nextSize;
    private final Neuron[] neurons;
    private double[] deltas;
    private double[][] weights;
    private final double[][] weightDeltas;
    private final double[][] weightDeltasPrev;
    private final double[] biases;
    private final double[] biasDeltas;
    private final double[] biasDeltasPrev;
    private int updates;
    private int biasUpdates;
    private double L2 = 0;

    public NeuronLayer(int size, int nextSize) {
        this.size = size;
        this.nextSize = nextSize;
        deltas = new double[size];
        weights = initializer.initWeights(new double[size][nextSize]);
        weightDeltas = new double[size][nextSize];
        weightDeltasPrev = new double[size][nextSize];
        biases = new double[nextSize];
        biasDeltas = new double[nextSize];
        biasDeltasPrev = new double[nextSize];
        neurons = new Neuron[size];
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron();
        }
    }

    public NeuronLayer(int size, int nextSize, Activation activation) {
        this(size, nextSize);
        this.activation = activation;
    }

    public NeuronLayer(int size, int nextSize, Activation activation, Initializer initializer) {
        this(size, nextSize, activation);
        this.initializer = initializer;
    }

    public NeuronLayer withBias(double initial) {
        Arrays.fill(biases, initial);
        return this;
    }

    public double getL2() {
        return L2;
    }

    public NeuronLayer withL2Regularization(double l2) {
        L2 = l2;
        return this;
    }

    public int getSize() {
        return size;
    }

    public int getNextSize() {
        return nextSize;
    }

    public double[] getDeltas() {
        return deltas;
    }

    public void setDeltas(double[] deltas) {
        this.deltas = deltas;
    }

    public double[] getBiases() {
        return biases;
    }

    public double[] getPrevBiasDeltas() {
        return biasDeltasPrev;
    }

    public double[][] getPrevWeightDeltas() {
        return weightDeltasPrev;
    }

    public Activation getActivation() {
        return activation;
    }

    public void saveWeightDeltas(double[][] weightDeltas) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                this.weightDeltas[i][j] += weightDeltas[i][j];
            }
        }
        updates++;
    }

    public void saveBiasDeltas(double[] biasDelta) {
        for (int i = 0; i < nextSize; i++) {
            this.biasDeltas[i] += biasDelta[i];
        }
        biasUpdates++;
    }

    public void update() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weightDeltasPrev[i][j] = (weightDeltas[i][j] / updates);
                if (L2 > 0){
                    double l2RegulizedWeight = weights[i][j] - weights[i][j] * L2;
                    weights[i][j] = l2RegulizedWeight + (weightDeltas[i][j] / updates);
                } else {
                    weights[i][j] = weights[i][j] + (weightDeltas[i][j] / updates);
                }
            }
        }
        for (int i = 0; i < nextSize; i++) {
            biasDeltasPrev[i] = (biasDeltas[i] / biasUpdates);
            biases[i] = biases[i] + (biasDeltas[i] / biasUpdates);
        }

        //clear
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weightDeltas[i][j] = 0;
            }
        }
        for (int i = 0; i < nextSize; i++) {
            biasDeltas[i] = 0;
        }
        updates = 0;
        biasUpdates = 0;
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

    public double[] calculateNetOutput() {
        double[] netOutput = new double[nextSize];
        for (int i = 0; i < nextSize; i++) {
            double total = 0.0;
            for (int j = 0; j < size; j++) {
                total += neurons[j].getValue() * weights[j][i];
            }
            netOutput[i] = total + biases[i];
        }
        return netOutput;
    }

    public double[] getOutput() {
        return Arrays.stream(neurons).mapToDouble(Neuron::getValue).toArray();
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }
}
