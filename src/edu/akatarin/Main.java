package edu.akatarin;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {
    private static Number fileToNumber(File file) throws IOException {
        List<String> lines = Files.readAllLines(file.toPath());
        int number = Integer.parseInt(lines.get(lines.size() - 1));
        double[] pixels = lines.stream().limit(lines.size() - 1)
                .flatMapToDouble(s -> {
                    String[] strings = s.split("\\t");
                    return Arrays.stream(strings).mapToDouble(val -> Double.parseDouble(val) / 256.0);
                }).toArray();
        return new Number(pixels, number);
    }

    public static void main(String[] args) throws Exception {
        File[] data = new File("F:\\mnist\\data").listFiles();
        if (data == null) return;
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        int epochs = 5000;
        double avgError;
        double total = 0;
        Random random = new Random();

        for (int i = 0; i < epochs; i++) {
            File file = data[random.nextInt(data.length)];
            Number number = fileToNumber(file);
            neuralNetwork.feedForward(number);
            neuralNetwork.backpropagation(number);
            double mse = neuralNetwork.getMSE(number);
            total += mse;
            avgError = (total) / epochs;
            System.out.print("\rEpoch: " + i + " MSE:" +mse+ " TotalError: " + avgError);
        }

        for (int i = 0; i < 10; i++) {
            Number testNumber = fileToNumber(data[random.nextInt(data.length)]);
            neuralNetwork.feedForward(testNumber);
            double[] output = neuralNetwork.getOutput();
            double max = Arrays.stream(output).max().orElse(Double.MIN_VALUE);
            for (int j = 0; j < output.length; j++) {
                if (max == output[j]) {
                    System.out.println("\nTest Number: "+testNumber.getValue()+" Predict:" + j);
                    break;
                }
            }
        }
    }
}
