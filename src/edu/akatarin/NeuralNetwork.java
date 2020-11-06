package edu.akatarin;

import java.util.Arrays;

public class NeuralNetwork {
    private CostFunction costFunction = CostFunction.MSE;//default
    private final static double LEARNING_RATE = 0.05;
    private final static double MOMENTUM = 0;//0.7-0.9
    private final static int INPUT_LAYER_SIZE = 784;
    private final static int HIDDEN_LAYERS_COUNT = 1;
    private final static int HIDDEN_LAYER_ONE_SIZE = 38;
    private final static int HIDDEN_LAYER_TWO_SIZE = 12;
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
    private final NeuronLayer[] hiddenLayers;
    private final NeuronLayer outputLayer;

    public NeuralNetwork() {
        inputLayer = new NeuronLayer(INPUT_LAYER_SIZE, HIDDEN_LAYER_ONE_SIZE,Activation.Sigmoid);
        hiddenLayers = new NeuronLayer[HIDDEN_LAYERS_COUNT]; //1 скрытый слой
        for (int i = 0; i < HIDDEN_LAYERS_COUNT; i++) {
            hiddenLayers[i] = new NeuronLayer(HIDDEN_LAYER_ONE_SIZE, OUTPUT_LAYER_SIZE, Activation.Sigmoid);
        }
        this.outputLayer = new NeuronLayer(OUTPUT_LAYER_SIZE, 0);
    }

    public NeuralNetwork(int inputLayerSize, int[] hiddenLayersSize, int outputLayerSize ) {
        if (inputLayerSize <= 0){
            throw new IllegalArgumentException("Illegal Input Layer capacity: "+inputLayerSize);
        }
        if (outputLayerSize <= 0){
            throw new IllegalArgumentException("Illegal Output Layer capacity: "+outputLayerSize);
        }
        int hiddenLayersCnt = hiddenLayersSize.length;
        if (hiddenLayersCnt <= 0) {
            throw new IllegalArgumentException("Illegal Hidden Layers count: "+hiddenLayersCnt);
        } else {
            for (int i = 0; i < hiddenLayersCnt; i++) {
                if (hiddenLayersSize[i] <= 0){
                    int layerPosition = i + 1;
                    throw new IllegalArgumentException("Illegal Hidden Layer "+layerPosition+" capacity: "+hiddenLayersSize[i]);
                }
            }
        }
        hiddenLayers = new NeuronLayer[hiddenLayersCnt];
        inputLayer = new NeuronLayer(inputLayerSize, hiddenLayersSize[0], Activation.Sigmoid);
        for (int i = 0; i < hiddenLayersCnt-1; i++) {
            hiddenLayers[i] = new NeuronLayer(hiddenLayersSize[i], hiddenLayersSize[i+1], Activation.Sigmoid);
        }
        int lastHiddenLayerPos = hiddenLayersCnt-1;
        hiddenLayers[lastHiddenLayerPos] = new NeuronLayer(hiddenLayersSize[lastHiddenLayerPos], outputLayerSize, Activation.Sigmoid);
        outputLayer = new NeuronLayer(outputLayerSize, 0, Activation.Sigmoid);
    }

    public NeuralNetwork(CostFunction costFunction) {
        this();
        this.costFunction = costFunction;
    }

    public NeuralNetwork(int inputLayerSize, int[] hiddenLayersSize, int outputLayerSize, CostFunction costFunction) {
        this(inputLayerSize,hiddenLayersSize,outputLayerSize);
        this.costFunction = costFunction;
    }

    //Mini-Batch Gradient Descent
    //обновляем веса входного и скрытого слоев на сумму DeltaW всех весов в пакете.
    public double trainBatch(Number[] batch) {
        //2-й скрытый слой
        double[][] hiddenTwo2OutDeltasSum = new double[HIDDEN_LAYER_TWO_SIZE][OUTPUT_LAYER_SIZE];
        double[] hiddenTwo2OutBiasDeltasSum = new double[OUTPUT_LAYER_SIZE];
        //1-й скрытый слой
        double[][] hiddenOne2hiddenTwoDeltasSum = new double[HIDDEN_LAYER_ONE_SIZE][HIDDEN_LAYER_TWO_SIZE];
        double[] hiddenOne2hiddenTwoBiasDeltasSum = new double[HIDDEN_LAYER_TWO_SIZE];
        //входной слой
        double[][] input2HiddenOneWeightsDeltasSum = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_ONE_SIZE];
        double[] input2HiddenOneBiasDeltasSum = new double[HIDDEN_LAYER_ONE_SIZE];
        double totalBatchMSE = 0;
//        for (Number number : batch) {
//            feedForward(number);
//            totalBatchMSE += getMSE(number);
//            //посчитаем ошибки выходного и скрытого слоев
//            double[] outputsError = calcOutputLayerError(number); //ошибка выходного слоя
//            double[] hiddenLayerTwoError = calcHiddenLayerTwoDelta(outputsError);
//            double[] hiddenLayerOneError = calcHiddenLayerOneDelta(hiddenLayerTwoError);
//
//            //подсчитаем градиент изменения весов 2-го скрытого слоя
//            double[][] hiddenTwo2OutDeltas = getHiddenLayerTwo2OutLayerWeightsDeltas(outputsError);
//            double[] hiddenTwo2OutBiasDeltas = getHiddenLayerTwo2OutLayerBiasDeltas(outputsError);
//            hiddenTwo2OutDeltasSum = sumDeltaW(hiddenTwo2OutDeltasSum, hiddenTwo2OutDeltas, HIDDEN_LAYER_TWO_SIZE, OUTPUT_LAYER_SIZE);
//            hiddenTwo2OutBiasDeltasSum = sumDeltaBias(hiddenTwo2OutBiasDeltasSum, hiddenTwo2OutBiasDeltas, OUTPUT_LAYER_SIZE);
//
//            //подсчитаем градиент изменения весов 1-го скрытого слоя
//            double[][] hiddenOne2HiddenTwoDeltas = getHiddenLayerOne2HiddenLayerTwoWeightsDeltas(hiddenLayerTwoError);
//            double[] hiddenOne2HiddenTwoBiasDeltas = getHiddenLayerOne2HiddenLayerTwoBiasDeltas(hiddenLayerTwoError);
//            hiddenOne2hiddenTwoDeltasSum = sumDeltaW(hiddenOne2hiddenTwoDeltasSum, hiddenOne2HiddenTwoDeltas, HIDDEN_LAYER_ONE_SIZE, HIDDEN_LAYER_TWO_SIZE);
//            hiddenOne2hiddenTwoBiasDeltasSum = sumDeltaBias(hiddenOne2hiddenTwoBiasDeltasSum, hiddenOne2HiddenTwoBiasDeltas, HIDDEN_LAYER_TWO_SIZE);
//
//            //подсчитаем градиент изменения весов входного слоя
//            double[][] input2HiddenOneWeightsDeltas = getInputLayer2HiddenLayerOneWeightsDeltas(hiddenLayerOneError);
//            double[] input2HiddenOneBiasDeltas = getInputLayer2HiddenLayerOneBiasDeltas(hiddenLayerOneError);
//            input2HiddenOneWeightsDeltasSum = sumDeltaW(input2HiddenOneWeightsDeltasSum, input2HiddenOneWeightsDeltas, INPUT_LAYER_SIZE, HIDDEN_LAYER_ONE_SIZE);
//            input2HiddenOneBiasDeltasSum = sumDeltaBias(input2HiddenOneBiasDeltasSum, input2HiddenOneBiasDeltas, HIDDEN_LAYER_ONE_SIZE);
//        }
//        //усредняем веса 2-го скрытого слоя
//        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
//            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
//                hiddenTwo2OutDeltasSum[i][j] *= 1.0 / batch.length;
//            }
//        }
//        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
//            hiddenTwo2OutBiasDeltasSum[i] *= 1.0 / batch.length;
//        }
//        //усредняем веса 1-го скрытого слоя
//        for (int i = 0; i < HIDDEN_LAYER_ONE_SIZE; i++) {
//            for (int j = 0; j < HIDDEN_LAYER_TWO_SIZE; j++) {
//                hiddenOne2hiddenTwoDeltasSum[i][j] *= 1.0 / batch.length;
//            }
//        }
//        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
//            hiddenOne2hiddenTwoBiasDeltasSum[i] *= 1.0 / batch.length;
//        }
//        //усредняем веса входного слоя
//        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
//            for (int j = 0; j < HIDDEN_LAYER_TWO_SIZE; j++) {
//                input2HiddenOneWeightsDeltasSum[i][j] *= 1.0 / batch.length;
//            }
//        }
//        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
//            input2HiddenOneBiasDeltasSum[i] *= 1.0 / batch.length;
//        }
//        updateHiddenLayerTwoWeights(hiddenTwo2OutDeltasSum, hiddenTwo2OutBiasDeltasSum);
//        updateHiddenLayerOneWeights(hiddenOne2hiddenTwoDeltasSum, hiddenOne2hiddenTwoBiasDeltasSum);
//        updateInputLayerWeights(input2HiddenOneWeightsDeltasSum,input2HiddenOneBiasDeltasSum);
        return totalBatchMSE;
    }

    private double[][] sumDeltaW( double[][] accumulator, double[][] delta, int size, int nextSize ){
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                accumulator[i][j] += delta[i][j];
            }
        }
        return accumulator;
    }

    private double[] sumDeltaBias(double[] accumulator, double[] delta, int size ){
        for (int i = 0; i < size; i++) {
            accumulator[i] += delta[i];
        }
        return accumulator;
    }

    public void feedForward(Number number) {
        double[] pixels = number.getPixels();
        inputLayer.setNeurons(pixels);
        NeuronLayer firstHiddenLayer = hiddenLayers[0];
        firstHiddenLayer.setNeurons(inputLayer.calculateOutput());
        if (hiddenLayers.length > 1){
            int hiddenLayersToFeed = hiddenLayers.length - 2;
            for (int i = 0; i <= hiddenLayersToFeed; i++) {
                NeuronLayer currentHiddenLayer = hiddenLayers[i];
                NeuronLayer nextHiddenLayer = hiddenLayers[i + 1];
                nextHiddenLayer.setNeurons(currentHiddenLayer.calculateOutput());
            }
        }
        NeuronLayer lastHiddenLayer = hiddenLayers[hiddenLayers.length-1];
        outputLayer.setNeurons(lastHiddenLayer.calculateOutput());
    }

    public double[] getOutput() {
        return Arrays.stream(outputLayer.getNeurons())
                .mapToDouble(Neuron::getValue).toArray();
    }

    public double getMSE(Number number) {
        double[] idealOut = ideals[number.getValue()];
        double[] currentOutput = getOutput();
        return costFunction.apply(idealOut, currentOutput);
    }

    //ошибка значений нейронов выходного слоя
    private void calcOutputLayerError(Number number) {
        double[] deltas = new double[OUTPUT_LAYER_SIZE];
        double[] idealOut = ideals[number.getValue()];
        double[] outputs = getOutput();
        double[] costFunctionDerivative = costFunction.applyDerivative(idealOut,outputs);
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            deltas[i] = costFunctionDerivative[i] * outputLayer.getActivation().applyDerivative(outputs[i]);
        }
        outputLayer.setDeltas(deltas);
    }

    //распространяем ошибку выходного слоя на скрытые слои
    private void calcHiddenLayerDelta(NeuronLayer currentLayer, NeuronLayer previousLayer) {
        int currentLayerSize = currentLayer.getSize();
        double[] deltas = new double[currentLayerSize];
        Neuron[] output = currentLayer.getNeurons();
        double[][] weights = currentLayer.getWeights();
        double[] previousLayerDelta = previousLayer.getDeltas();
        for (int i = 0; i < currentLayerSize; i++) {
            double error = 0;
            double[] neuronWeights = weights[i];
            for (int j = 0; j < previousLayerDelta.length; j++) {
                error += previousLayerDelta[j] * neuronWeights[j];
            }
            deltas[i] = error * currentLayer.getActivation().applyDerivative(output[i].getValue());
        }
        currentLayer.setDeltas(deltas);
    }

    //    изменение веса синапса равно коэффициенту скорости обучения, умноженному на градиент этого веса,
    //    прибавить момент умноженный на предыдущее изменение этого веса (на 1-ой итерации равно 0)
    private void updateWeightsAndBiases(NeuronLayer currentLayer, NeuronLayer nextLayer){
        int size = currentLayer.getSize();
        int nextSize = currentLayer.getNextSize();
        double[][] currentWeightsDelta = new double[size][nextSize];
        double[][] previousWeightsDelta = currentLayer.getWeightDeltas();
        double[] currentBiasDelta = new double[nextSize];
        double[] previousBiasDelta = currentLayer.getBiasDeltas();
        Neuron[] neurons = currentLayer.getNeurons();
        double[] nextLayerError = nextLayer.getDeltas();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                double gradient = neurons[i].getValue() * nextLayerError[j];
                currentWeightsDelta[i][j] = (gradient * LEARNING_RATE) + (MOMENTUM * previousWeightsDelta[i][j]);
            }
        }
        for (int i = 0; i < nextSize; i++) {
            currentBiasDelta[i] = (nextLayerError[i] * LEARNING_RATE) + (MOMENTUM * previousBiasDelta[i]);
        }
        currentLayer.updateWeights(currentWeightsDelta);
        currentLayer.updateBias(currentBiasDelta);
    }

    //Stochastic Gradient Descent
    //Веса обновляются в real-time, сразу после того как посчитано DeltaW
    public void backpropagation(Number number) {
        calcOutputLayerError(number); //ошибка выходного слоя
        //распространяем ошибку на скрытые слои
        NeuronLayer lastHiddenLayer = hiddenLayers[hiddenLayers.length-1];
        calcHiddenLayerDelta(lastHiddenLayer, outputLayer);
        if (hiddenLayers.length > 1) {
            int hiddenLayersToPropagate = hiddenLayers.length - 2;
            for (int i = hiddenLayersToPropagate; i >= 0; i--) {
                NeuronLayer currentHiddenLayer = hiddenLayers[i];
                NeuronLayer previousHiddenLayer = hiddenLayers[i + 1];
                calcHiddenLayerDelta(currentHiddenLayer, previousHiddenLayer);
            }
        }

        //подсчитаем градиент изменения весов скрытого слоя и сразу же обновим веса
        NeuronLayer firstHiddenLayer = hiddenLayers[0];
        updateWeightsAndBiases(inputLayer, firstHiddenLayer);
        if (hiddenLayers.length > 1) {
            int hiddenLayersToUpdate = hiddenLayers.length - 2;
            for (int i = 0; i <= hiddenLayersToUpdate; i++) {
                NeuronLayer currentHiddenLayer = hiddenLayers[i];
                NeuronLayer nextHiddenLayer = hiddenLayers[i + 1];
                updateWeightsAndBiases(currentHiddenLayer, nextHiddenLayer);
            }
        }
        updateWeightsAndBiases(lastHiddenLayer, outputLayer);
    }
}
