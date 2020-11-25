package edu.akatarin;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    private static File[] data;
    private static List<Number> numbers;
    private static final double[][] numbersToTrain = {
            {0, 0, 0, 0}, //0
            {0, 0, 0, 1}, //1
            {0, 0, 1, 0}, //2
            {0, 0, 1, 1}, //3
            {0, 1, 0, 0}, //4
            {0, 1, 0, 1}, //5
            {0, 1, 1, 0}, //6
            {0, 1, 1, 1}, //7
            {1, 0, 0, 0}, //8
            {1, 0, 0, 1}, //9
    };

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

    private final static int EPOCH_LIMIT = 10_000;

    private static Number fileToNumber(File file) {
        try {
            List<String> lines = Files.readAllLines(file.toPath());
            int number = Integer.parseInt(lines.get(lines.size() - 1));
            double[] pixels = lines.stream().limit(lines.size() - 1)
                    .flatMapToDouble(s -> {
                        String[] strings = s.split("\\t");
                        return Arrays.stream(strings).mapToDouble(val -> Double.parseDouble(val) / 255.0);
                    }).toArray();
            return new Number(pixels, number);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static void testMNISTStochasticGD() {
        NeuronLayer input = new NeuronLayer(784, 800, Activation.Sigmoid, Initializer.XAVIER_NORMAL);
        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.neuralNetworkBuilder()
                .withInputLayer(input)
                .addHiddenLayer(new NeuronLayer(800, 10, Activation.Sigmoid, Initializer.XAVIER_NORMAL))
                .withOutputLayer(new NeuronLayer(10, 0, Activation.Softmax))
                .build();
        neuralNetwork.setCostFunction(CostFunction.CROSS_ENTROPY);
        neuralNetwork.setOptimizer(new Optimizer.GradientDescent(0.01));
        neuralNetwork.setMomentum(0.7);
        Collections.shuffle(numbers);
        for (int i = 0; i < 3; i++) {
            int epoch = 0;
            for (Number number : numbers) {
                epoch++;
                double[] idealOut = ideals[number.getValue()];
                neuralNetwork.feedForward(number);
                neuralNetwork.backpropagation(idealOut);
                double mse = neuralNetwork.getCost(idealOut);
                if (epoch % 300 == 0) {
                    System.out.print("\rEpoch:" + epoch + " Number: " + number.getValue() + " ERROR: " + mse);
                }
            }
        }
        testMNISTNumberRecognition(neuralNetwork);
    }

    private static void trainMiniBatchGD(int batchSize) {
        NeuronLayer input = new NeuronLayer(784, 800, Activation.Sigmoid, Initializer.XAVIER_NORMAL);
        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.neuralNetworkBuilder()
                .withInputLayer(input)
                .addHiddenLayer(new NeuronLayer(800, 10, Activation.Sigmoid, Initializer.XAVIER_NORMAL))
                .withOutputLayer(new NeuronLayer(10, 0, Activation.Softmax, Initializer.XAVIER_NORMAL))
                .build();
        neuralNetwork.setCostFunction(CostFunction.CROSS_ENTROPY);
        neuralNetwork.setOptimizer(new Optimizer.GradientDescent(0.5));
        neuralNetwork.setMomentum(0.7);
        Collections.shuffle(numbers);

        int batchCount = numbers.size() / batchSize;
        double[][] expectedOutputs = new double[batchSize][10];
        int epoch = 0;
        double batchError;
        do {
            epoch++;
            for (int i = 0; i < batchCount; i++) {
                int fromIndex = i * batchSize;
                int toIndex = Math.min(numbers.size(), (i + 1) * batchSize);
                List<Number> batch = numbers.subList(fromIndex, toIndex);
                for (int j = 0; j < batchSize; j++) {
                    expectedOutputs[j] = ideals[batch.get(j).getValue()];
                }
                double totalBatchError = neuralNetwork.trainBatch(batch.toArray(new Number[batchSize]), expectedOutputs);
                batchError = totalBatchError / batchSize;
                System.out.print("\rEpoch: " + epoch + " Batch: " + i + " of " + batchCount + " ERROR: " + totalBatchError + " AVG ERR: " + batchError);
            }
        } while ( epoch <= 3);
        testMNISTNumberRecognition(neuralNetwork);
    }

    private static void testMNISTNumberRecognition(NeuralNetwork nn) {
        double overall = 0;
        Random random = new Random();
        for (int i = 0; i < 100; i++) {
            Number testNumber = numbers.get(random.nextInt(numbers.size()));
            double[] expected = ideals[testNumber.getValue()];
            nn.feedForward(testNumber);
            double[] output = nn.getOutput();
            double max = Arrays.stream(output).max().orElse(Double.MIN_VALUE);
            boolean passed = false;
            for (int j = 0; j < output.length; j++) {
                if (max == output[j]) {
                    if (testNumber.getValue() == j) {
                        passed = true;
                        overall++;
                    }
                    break;
                }
            }
            System.out.println("Test : " + testNumber.getValue() + " " + passed + " LOSS: " + nn.getCost(expected));
        }
        System.out.println("Overall score: " + overall / 100.0);
    }

    //Тест на основе алгоритма https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    private static void testNetworkStepByStep() {
        double[][][] initialweights = {
                {
                        {0.15, 0.25}, //w1,w3
                        {0.20, 0.30} //w2,w4
                },
                {
                        {0.40, 0.50},//w5,w7
                        {0.45, 0.55}//w6,w8
                }
        };
        NeuronLayer input = new NeuronLayer(2, 2, Activation.Sigmoid, Initializer.MANUAL);
        input.setWeights(initialweights[0]);
        input.withBias(0.35);

        NeuronLayer hidden = new NeuronLayer(2, 2, Activation.Sigmoid, Initializer.MANUAL);
        hidden.setWeights(initialweights[1]);
        hidden.withBias(0.60);

        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.neuralNetworkBuilder()
                .withInputLayer(input)
                .addHiddenLayer(hidden)
                .withOutputLayer(new NeuronLayer(2, 0))
                .build();
        neuralNetwork.setCostFunction(CostFunction.HALF_QUADRATIC);

        double[] example = {0.05, 0.10};
        double[] expected = {0.01, 0.99};

        Number number = new Number(example, 0);
        neuralNetwork.feedForward(number);
        System.out.println("Initial COST: " + neuralNetwork.getCost(expected));
        System.out.println("-===== LEARN =====-");
        int epoch = 0;
        do {
            epoch++;
            neuralNetwork.backpropagation(expected);
            neuralNetwork.feedForward(number);
            double cost = neuralNetwork.getCost(expected);
            if (epoch % 60 == 0) {
                System.out.print("\rEpoch: " + epoch + " COST: " + cost);
                if (cost <= 0.000000001 || epoch > EPOCH_LIMIT) {
                    System.out.println();
                    System.out.println("OUT: " + Arrays.toString(neuralNetwork.getOutput()));
                    break;
                }
            }
        } while (true);
    }

    private static void testNetwork() {
        NeuronLayer input = new NeuronLayer(4, 6, Activation.Identity, Initializer.XAVIER_NORMAL);
        input.withBias(0.5);

        NeuronLayer hidden = new NeuronLayer(6, 10, Activation.Sigmoid, Initializer.XAVIER_NORMAL);
        hidden.withBias(0.5);

        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.neuralNetworkBuilder()
                .withInputLayer(input)
                .addHiddenLayer(hidden)
                .withOutputLayer(new NeuronLayer(10, 0, Activation.Softmax))
                .build();
        neuralNetwork.setCostFunction(CostFunction.CROSS_ENTROPY);
        neuralNetwork.setOptimizer(new Optimizer.GradientDescent(0.01));
        neuralNetwork.setMomentum(0.7);

        Random random = new Random();
        int initialIdx = random.nextInt(numbersToTrain.length);
        Number initialNumber = new Number(numbersToTrain[initialIdx], initialIdx);
        neuralNetwork.feedForward(initialNumber);
        System.out.println("Initial COST: " + neuralNetwork.getCost(ideals[initialNumber.getValue()]));
        System.out.println("-===== LEARN =====-");
        int epoch = 0;
        do {
            epoch++;
            double totalCost = 0;
            for (int j = 0; j < numbersToTrain.length; j++) {
                Number number = new Number(numbersToTrain[j], j);
                double[] expected = ideals[number.getValue()];
                neuralNetwork.feedForward(number);
                neuralNetwork.backpropagation(expected);
                double cost = neuralNetwork.getCost(expected);
                totalCost += cost;
            }
            if (epoch % 10 == 0) {
                double avgCost = totalCost / numbersToTrain.length;
                System.out.print("\rEpoch: " + epoch + " Average COST: " + avgCost);
                if (avgCost <= 0.01 || epoch > EPOCH_LIMIT) {
                    System.out.println();
                    break;
                }
            }
        } while (true);
        double overall = 0;
        for (int i = 0; i < numbersToTrain.length; i++) {
            Number testNumber = new Number(numbersToTrain[i], i);
            double[] expected = ideals[testNumber.getValue()];
            neuralNetwork.feedForward(testNumber);
            double[] output = neuralNetwork.getOutput();
            double max = Arrays.stream(output).max().orElse(Double.MIN_VALUE);
            boolean passed = false;
            for (int j = 0; j < output.length; j++) {
                if (max == output[j]) {
                    if (testNumber.getValue() == j) {
                        passed = true;
                        overall++;
                    }
                    break;
                }
            }
            System.out.println("Test : " + testNumber.getValue() + " " + passed + " LOSS: " + neuralNetwork.getCost(expected));
        }
        System.out.println("Overall score: " + overall / numbersToTrain.length);
    }

    private static void testNetworkBatch() {
        NeuronLayer input = new NeuronLayer(4, 6, Activation.Identity, Initializer.XAVIER_NORMAL);
        input.withBias(0.5);

        NeuronLayer hidden = new NeuronLayer(6, 10, Activation.Sigmoid, Initializer.XAVIER_NORMAL);
        hidden.withBias(0.5);

        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.neuralNetworkBuilder()
                .withInputLayer(input)
                .addHiddenLayer(hidden)
                .withOutputLayer(new NeuronLayer(10, 0, Activation.Softmax))
                .build();
        neuralNetwork.setCostFunction(CostFunction.CROSS_ENTROPY);
        neuralNetwork.setOptimizer(new Optimizer.GradientDescent(0.01));
        neuralNetwork.setMomentum(0.7);

        Random random = new Random();
        int initialIdx = random.nextInt(numbersToTrain.length);
        Number initialNumber = new Number(numbersToTrain[initialIdx], initialIdx);
        neuralNetwork.feedForward(initialNumber);
        System.out.println("Initial COST: " + neuralNetwork.getCost(ideals[initialNumber.getValue()]));
        System.out.println("-===== LEARN =====-");
        int epoch = 0;
        do {
            epoch++;
            Number[] numbers = new Number[numbersToTrain.length];
            for (int j = 0; j < numbersToTrain.length; j++) {
                numbers[j] = new Number(numbersToTrain[j], j);
            }
            double averageBatchError = neuralNetwork.trainBatch(numbers, ideals) / numbers.length;
            if (epoch % 10 == 0) {
                System.out.print("\rEpoch: " + epoch + " ERROR: " + averageBatchError);
                if (averageBatchError <= 0.01 || epoch > EPOCH_LIMIT) {
                    System.out.println();
                    break;
                }
            }
        } while (true);
        double overall = 0;
        for (int i = 0; i < numbersToTrain.length; i++) {
            Number testNumber = new Number(numbersToTrain[i], i);
            double[] expected = ideals[testNumber.getValue()];
            neuralNetwork.feedForward(testNumber);
            double[] output = neuralNetwork.getOutput();
            double max = Arrays.stream(output).max().orElse(Double.MIN_VALUE);
            boolean passed = false;
            for (int j = 0; j < output.length; j++) {
                if (max == output[j]) {
                    if (testNumber.getValue() == j) {
                        passed = true;
                        overall++;
                    }
                    break;
                }
            }
            System.out.println("Test : " + testNumber.getValue() + " " + passed + " LOSS: " + neuralNetwork.getCost(expected));
        }
        System.out.println("Overall score: " + overall / numbersToTrain.length);
    }

    private static void loadMNISTDB(){
        data = new File("F:\\mnist\\data").listFiles();
        if (data == null) System.exit(0);
        numbers = Arrays.stream(data).parallel().map(Main::fileToNumber)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    public static void main(String[] args) {
        System.out.println("Test 1");
        testNetworkStepByStep();
        System.out.println("Test 2");
        testNetwork();
        System.out.println("Test 3");
        testNetworkBatch();

        loadMNISTDB();
        System.out.println("Test MNIST 1");
        testMNISTStochasticGD();
        System.out.println("Test MNIST 2");
        trainMiniBatchGD(32);
    }
}
