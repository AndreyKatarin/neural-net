package edu.akatarin;

import java.util.Arrays;

public class NeuralNetwork {
    private final static double LEARNING_RATE = 0.05;
    private final static double MOMENTUM = 0.1;
    private final static int INPUT_LAYER_SIZE = 784;
    private final static int HIDDEN_LAYER_ONE_SIZE = 300;
    private final static int HIDDEN_LAYER_TWO_SIZE = 50;
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
    private final NeuronLayer hiddenLayerOne;
    private final NeuronLayer hiddenLayerTwo;
    private final NeuronLayer outputLayer;

    public NeuralNetwork() {
        this.inputLayer = new NeuronLayer(INPUT_LAYER_SIZE, HIDDEN_LAYER_ONE_SIZE);
        this.hiddenLayerOne = new NeuronLayer(HIDDEN_LAYER_ONE_SIZE, HIDDEN_LAYER_TWO_SIZE);
        this.hiddenLayerTwo = new NeuronLayer(HIDDEN_LAYER_TWO_SIZE, OUTPUT_LAYER_SIZE);
        this.outputLayer = new NeuronLayer(OUTPUT_LAYER_SIZE, 0);
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
        for (Number number : batch) {
            feedForward(number);
            totalBatchMSE += getMSE(number);
            //посчитаем ошибки выходного и скрытого слоев
            double[] outputsError = calcOutputLayerError(number); //ошибка выходного слоя
            double[] hiddenLayerTwoError = calcHiddenLayerTwoDelta(outputsError);
            double[] hiddenLayerOneError = calcHiddenLayerOneDelta(hiddenLayerTwoError);

            //подсчитаем градиент изменения весов 2-го скрытого слоя
            double[][] hiddenTwo2OutDeltas = getHiddenLayerTwo2OutLayerWeightsDeltas(outputsError);
            double[] hiddenTwo2OutBiasDeltas = getHiddenLayerTwo2OutLayerBiasDeltas(outputsError);
            hiddenTwo2OutDeltasSum = sumDeltaW(hiddenTwo2OutDeltasSum, hiddenTwo2OutDeltas, HIDDEN_LAYER_TWO_SIZE, OUTPUT_LAYER_SIZE);
            hiddenTwo2OutBiasDeltasSum = sumDeltaBias(hiddenTwo2OutBiasDeltasSum, hiddenTwo2OutBiasDeltas, OUTPUT_LAYER_SIZE);

            //подсчитаем градиент изменения весов 1-го скрытого слоя
            double[][] hiddenOne2HiddenTwoDeltas = getHiddenLayerOne2HiddenLayerTwoWeightsDeltas(hiddenLayerTwoError);
            double[] hiddenOne2HiddenTwoBiasDeltas = getHiddenLayerOne2HiddenLayerTwoBiasDeltas(hiddenLayerTwoError);
            hiddenOne2hiddenTwoDeltasSum = sumDeltaW(hiddenOne2hiddenTwoDeltasSum, hiddenOne2HiddenTwoDeltas, HIDDEN_LAYER_ONE_SIZE, HIDDEN_LAYER_TWO_SIZE);
            hiddenOne2hiddenTwoBiasDeltasSum = sumDeltaBias(hiddenOne2hiddenTwoBiasDeltasSum, hiddenOne2HiddenTwoBiasDeltas, HIDDEN_LAYER_TWO_SIZE);

            //подсчитаем градиент изменения весов входного слоя
            double[][] input2HiddenOneWeightsDeltas = getInputLayer2HiddenLayerOneWeightsDeltas(hiddenLayerOneError);
            double[] input2HiddenOneBiasDeltas = getInputLayer2HiddenLayerOneBiasDeltas(hiddenLayerOneError);
            input2HiddenOneWeightsDeltasSum = sumDeltaW(input2HiddenOneWeightsDeltasSum, input2HiddenOneWeightsDeltas, INPUT_LAYER_SIZE, HIDDEN_LAYER_ONE_SIZE);
            input2HiddenOneBiasDeltasSum = sumDeltaBias(input2HiddenOneBiasDeltasSum, input2HiddenOneBiasDeltas, HIDDEN_LAYER_ONE_SIZE);
        }
        //усредняем веса 2-го скрытого слоя
        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                hiddenTwo2OutDeltasSum[i][j] *= 1.0 / batch.length;
            }
        }
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            hiddenTwo2OutBiasDeltasSum[i] *= 1.0 / batch.length;
        }
        //усредняем веса 1-го скрытого слоя
        for (int i = 0; i < HIDDEN_LAYER_ONE_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_TWO_SIZE; j++) {
                hiddenOne2hiddenTwoDeltasSum[i][j] *= 1.0 / batch.length;
            }
        }
        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
            hiddenOne2hiddenTwoBiasDeltasSum[i] *= 1.0 / batch.length;
        }
        //усредняем веса входного слоя
        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_TWO_SIZE; j++) {
                input2HiddenOneWeightsDeltasSum[i][j] *= 1.0 / batch.length;
            }
        }
        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
            input2HiddenOneBiasDeltasSum[i] *= 1.0 / batch.length;
        }
        updateHiddenLayerTwoWeights(hiddenTwo2OutDeltasSum, hiddenTwo2OutBiasDeltasSum);
        updateHiddenLayerOneWeights(hiddenOne2hiddenTwoDeltasSum, hiddenOne2hiddenTwoBiasDeltasSum);
        updateInputLayerWeights(input2HiddenOneWeightsDeltasSum,input2HiddenOneBiasDeltasSum);
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
        hiddenLayerOne.setNeurons(inputLayer.calculateOutput());
        hiddenLayerTwo.setNeurons(hiddenLayerOne.calculateOutput());
        outputLayer.setNeurons(hiddenLayerTwo.calculateOutput());
    }

    public double[] getOutput() {
        return Arrays.stream(outputLayer.getNeurons())
                .mapToDouble(Neuron::getValue).toArray();
    }

    //Cost function: Mean square error, C = 1/n * ∑(y−exp)^2
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

    //ошибка значений нейронов выходного слоя
    private double[] calcOutputLayerError(Number number) {
        double[] deltas = new double[OUTPUT_LAYER_SIZE];
        double[] idealOut = ideals[number.getValue()];
        double[] outputs = getOutput();
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            double error = idealOut[i] - outputs[i];
            deltas[i] = 2 * error * sigmDx(outputs[i]);
//            double mseDerivative = 2 * (outputs[i] - idealOut[i]);
//            deltas[i] = mseDerivative * sigmDx(outputs[i]);
        }
        return deltas;
    }

    //распространяем ошибку выходного слоя на 2-й скрытый слой
    private double[] calcHiddenLayerTwoDelta(double[] outputsDelta) {
        double[] deltas = new double[HIDDEN_LAYER_TWO_SIZE];
        Neuron[] output = hiddenLayerTwo.getNeurons();
        double[][] weights = hiddenLayerTwo.getWeights();
        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
            double error = 0;
            double[] hiddenNeuronWeights = weights[i];
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                error += outputsDelta[j] * hiddenNeuronWeights[j];
            }
            deltas[i] = error * sigmDx(output[i].getValue());
        }
        return deltas;
    }

    //распространяем ошибку выходного слоя на 1-й скрытый слой
    private double[] calcHiddenLayerOneDelta(double[] outputsDelta) {
        double[] deltas = new double[HIDDEN_LAYER_ONE_SIZE];
        Neuron[] output = hiddenLayerOne.getNeurons();
        double[][] weights = hiddenLayerOne.getWeights();
        for (int i = 0; i < HIDDEN_LAYER_ONE_SIZE; i++) {
            double error = 0;
            double[] hiddenNeuronWeights = weights[i];
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                error += outputsDelta[j] * hiddenNeuronWeights[j];
            }
            deltas[i] = error * sigmDx(output[i].getValue());
        }
        return deltas;
    }

    private void updateInputLayerWeights(double[][] deltas, double[] biasDelta) {
        inputLayer.updateWeights(deltas);
        inputLayer.updateBias(biasDelta);
    }

    private void updateHiddenLayerTwoWeights(double[][] deltas, double[] biasDelta) {
        hiddenLayerTwo.updateWeights(deltas);
        hiddenLayerTwo.updateBias(biasDelta);
    }

    private void updateHiddenLayerOneWeights(double[][] deltas, double[] biasDelta) {
        hiddenLayerOne.updateWeights(deltas);
        hiddenLayerOne.updateBias(biasDelta);
    }

    //    изменение веса синапса равно коэффициенту скорости обучения, умноженному на градиент этого веса,
    //    прибавить момент умноженный на предыдущее изменение этого веса (на 1-ой итерации равно 0)
    private double[][] getHiddenLayerTwo2OutLayerWeightsDeltas(double[] outputsError){
        double[][] deltas = new double[HIDDEN_LAYER_TWO_SIZE][OUTPUT_LAYER_SIZE];
        Neuron[] neurons = hiddenLayerTwo.getNeurons();
        double[][] prevWeightDeltas = hiddenLayerTwo.getWeightDeltas();
        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                double gradient = neurons[i].getValue() * outputsError[j];
                deltas[i][j] = (gradient * LEARNING_RATE) + (MOMENTUM * prevWeightDeltas[i][j]);
            }
        }
        return deltas;
    }

    private double[][] getHiddenLayerOne2HiddenLayerTwoWeightsDeltas(double[] hiddenLayerTwoDelta){
        double[][] deltas = new double[HIDDEN_LAYER_ONE_SIZE][HIDDEN_LAYER_TWO_SIZE];
        Neuron[] neurons = hiddenLayerOne.getNeurons();
        double[][] prevWeightDeltas = hiddenLayerOne.getWeightDeltas();
        for (int i = 0; i < HIDDEN_LAYER_ONE_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_TWO_SIZE; j++) {
                double gradient = neurons[i].getValue() * hiddenLayerTwoDelta[j];
                deltas[i][j] = (gradient * LEARNING_RATE) + (MOMENTUM * prevWeightDeltas[i][j]);
            }
        }
        return deltas;
    }

    private double[][] getInputLayer2HiddenLayerOneWeightsDeltas(double[] hiddenL1WeightsDelta) {
        double[][] deltas = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_ONE_SIZE];
        Neuron[] neurons = inputLayer.getNeurons();
        double[][] prevWeightDeltas = inputLayer.getWeightDeltas();
        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_ONE_SIZE; j++) {
                double gradient = neurons[i].getValue() * hiddenL1WeightsDelta[j];
                deltas[i][j] = (gradient * LEARNING_RATE) + (MOMENTUM * prevWeightDeltas[i][j]);
            }
        }
        return deltas;
    }

    private double[] getHiddenLayerTwo2OutLayerBiasDeltas(double[] outputsDelta){
        double[] biasDelta = new double[OUTPUT_LAYER_SIZE];
        double[] prevBiasDeltas = hiddenLayerTwo.getBiasDeltas();
        for (int i = 0; i < HIDDEN_LAYER_TWO_SIZE; i++) {
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                biasDelta[j] = (outputsDelta[j] * LEARNING_RATE) + (MOMENTUM * prevBiasDeltas[j]);
            }
        }
        return biasDelta;
    }

    private double[] getHiddenLayerOne2HiddenLayerTwoBiasDeltas(double[] hiddenLayerTwoDelta){
        double[] biasDelta = new double[HIDDEN_LAYER_TWO_SIZE];
        double[] prevBiasDeltas = hiddenLayerOne.getBiasDeltas();
        for (int i = 0; i < HIDDEN_LAYER_ONE_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_TWO_SIZE; j++) {
                biasDelta[j] = (hiddenLayerTwoDelta[j] * LEARNING_RATE) + (MOMENTUM * prevBiasDeltas[j]);
            }
        }
        return biasDelta;
    }

    private double[] getInputLayer2HiddenLayerOneBiasDeltas(double[] hiddenL1WeightsDelta) {
        double[] biasDelta = new double[HIDDEN_LAYER_ONE_SIZE];
        double[] prevBiasDeltas = inputLayer.getBiasDeltas();
        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_ONE_SIZE; j++) {
                biasDelta[j] = (hiddenL1WeightsDelta[j] * LEARNING_RATE) + (MOMENTUM * prevBiasDeltas[j]);
            }
        }
        return biasDelta;
    }

    //Stochastic Gradient Descent
    //Веса обновляются в real-time, сразу после того как посчитано DeltaW
    public void backpropagation(Number number) {
        double[] outputsError = calcOutputLayerError(number); //ошибка выходного слоя
        double[] hiddenLayerTwoError = calcHiddenLayerTwoDelta(outputsError); //ошибка 2-го скрытого слоя
        double[] hiddenLayerOneError = calcHiddenLayerOneDelta(hiddenLayerTwoError); //ошибка 1-го скрытого слоя

        //подсчитаем градиент изменения весов 2-го скрытого слоя и сразу же обновим веса
        double[][] hiddenTwo2OutDeltas = getHiddenLayerTwo2OutLayerWeightsDeltas(outputsError);
        double[] hiddenTwo2OutBiasDeltas = getHiddenLayerTwo2OutLayerBiasDeltas(outputsError);
        updateHiddenLayerTwoWeights(hiddenTwo2OutDeltas, hiddenTwo2OutBiasDeltas);

        //подсчитаем градиент изменения весов 1-го скрытого слоя и сразу же обновим веса
        double[][] hiddenOne2HiddenTwoDeltas = getHiddenLayerOne2HiddenLayerTwoWeightsDeltas(hiddenLayerTwoError);
        double[] hiddenOne2HiddenTwoBiasDeltas = getHiddenLayerOne2HiddenLayerTwoBiasDeltas(hiddenLayerTwoError);
        updateHiddenLayerOneWeights(hiddenOne2HiddenTwoDeltas, hiddenOne2HiddenTwoBiasDeltas);

        //подсчитаем градиент изменения весов входного слоя и сразу же обновим веса
        double[][] input2HiddenOneWeightsDeltas = getInputLayer2HiddenLayerOneWeightsDeltas(hiddenLayerOneError);
        double[] input2HiddenOneBiasDeltas = getInputLayer2HiddenLayerOneBiasDeltas(hiddenLayerOneError);
        updateInputLayerWeights(input2HiddenOneWeightsDeltas, input2HiddenOneBiasDeltas);
    }
}
