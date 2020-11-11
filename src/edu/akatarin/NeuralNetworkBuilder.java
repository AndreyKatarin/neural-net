package edu.akatarin;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkBuilder {

    private NeuronLayer inputLayer;
    private final List<NeuronLayer> hiddenLayers = new ArrayList<>();
    private NeuronLayer outputLayer;

    private NeuralNetworkBuilder() {}

    public static NeuralNetworkBuilder neuralNetworkBuilder(){
        return new NeuralNetworkBuilder();
    }

    public NeuralNetworkBuilder withInputLayer(NeuronLayer layer) {
        this.inputLayer = layer;
        return this;
    }

    public NeuralNetworkBuilder withOutputLayer(NeuronLayer layer) {
        this.outputLayer = layer;
        return this;
    }

    public NeuralNetworkBuilder addHiddenLayer(NeuronLayer layer){
        hiddenLayers.add(layer);
        return this;
    }

    public NeuralNetwork build() {
        NeuronLayer[] hiddenLayers = new NeuronLayer[this.hiddenLayers.size()];
        hiddenLayers = this.hiddenLayers.toArray(hiddenLayers);
        return new NeuralNetwork(inputLayer, hiddenLayers, outputLayer);
    }
}
