package edu.akatarin;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {
    private static File[] data;
    private static Number fileToNumber(File file) throws IOException {
        List<String> lines = Files.readAllLines(file.toPath());
        int number = Integer.parseInt(lines.get(lines.size() - 1));
        double[] pixels = lines.stream().limit(lines.size() - 1)
                .flatMapToDouble(s -> {
                    String[] strings = s.split("\\t");
                    return Arrays.stream(strings).mapToDouble(val -> Double.parseDouble(val) / 255.0);
                }).toArray();
        return new Number(pixels, number);
    }

    private static void trainStochasticGD(NeuralNetwork neuralNetwork) throws IOException {
        int epochs = 100;
        double avgError;
        Random random = new Random();
        for (int i = 0; i < epochs; i++) {
            double total = 0;
            for (int j = 0; j < data.length; j++) {
                File file = data[random.nextInt(data.length)];
                Number number = fileToNumber(file);
                neuralNetwork.feedForward(number);
                neuralNetwork.backpropagation(number);
                double mse = neuralNetwork.getMSE(number);
                total += mse;
            }
            avgError = (total) / data.length;
            System.out.print("\rEpoch: " + i + " Avg.MSE:" +avgError);
        }
    }

    private static void trainMiniBatchGD(NeuralNetwork neuralNetwork, int batchSize) throws IOException {
        int epochs = data.length / batchSize;
        Random random = new Random();
        for (int e = 0; e < 100; e++) {
            double total =0;
            for (int i = 0; i < epochs; i++) {
                Number[] batch = new Number[batchSize];
                for (int j = 0; j < batchSize; j++) {
                    File file = data[random.nextInt(data.length)];
                    batch[j] = fileToNumber(file);
                }
                double MAE_ERROR = neuralNetwork.trainBatch(batch) / batchSize;
                total += MAE_ERROR;
            }
            double ERR = (1.0 / batchSize) * total;
            System.out.print("\rEpoch: " + e + " MAE_ERROR:" +ERR+" Total: "+total);
        }
    }

    public static void main(String[] args) throws Exception {
        data = new File("F:\\mnist\\data").listFiles();
        if (data == null) return;
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        trainStochasticGD(neuralNetwork);
        //trainMiniBatchGD(neuralNetwork, 200);
        double overall = 0;
        Random random = new Random();
        for (int i = 0; i < 100; i++) {
            Number testNumber = fileToNumber(data[random.nextInt(data.length)]);
            neuralNetwork.feedForward(testNumber);
            double[] output = neuralNetwork.getOutput();
            double max = Arrays.stream(output).max().orElse(Double.MIN_VALUE);
            for (int j = 0; j < output.length; j++) {
                if (max == output[j]) {
                    if (testNumber.getValue()==j) overall++;
                    break;
                }
            }
        }
        System.out.print("\nOverall score: "+overall/100.0);
    }
}
