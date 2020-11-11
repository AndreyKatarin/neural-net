package edu.akatarin;

import java.util.Random;
import java.util.function.UnaryOperator;

public class Initializer {
    private final String name;
    private final UnaryOperator<double[][]> initializer;

    private static final Random random = new Random();

    private Initializer(String name, UnaryOperator<double[][]> initializer) {
        this.name = name;
        this.initializer = initializer;
    }

    public double[][] initWeights(double[][] weights) {
        return initializer.apply(weights);
    }

    public String getName() {
        return name;
    }

    //по умолчанию
    public static Initializer RANDOM_GAUSSIAN = new Initializer(
            "RANDOM_GAUSSIAN",
            (weights) -> {
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        weights[i][j] = random.nextGaussian();
                    }
                }
                return weights;
            }
    );

    public static Initializer XAVIER_UNIFORM = new Initializer(
            "XAVIER_UNIFORM",
            (weights) -> {
                double factor = Math.sqrt(6.0 / (weights.length + weights[0].length));
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        weights[i][j] = ((random.nextDouble() * 2)-1) * factor;
                    }
                }
                return weights;
            }
    );

    public static Initializer XAVIER_NORMAL = new Initializer(
            "XAVIER_NORMAL",
            (weights) -> {
                double factor = Math.sqrt(2.0 / (weights.length + weights[0].length));
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        weights[i][j] = random.nextGaussian() * factor;
                    }
                }
                return weights;
            }
    );

    public static Initializer MANUAL = new Initializer(
            "MANUAL",
            (weights) -> weights
    );

    //ReLU Leaky_ReLU
    public static Initializer HE_NORMAL = new Initializer(
            "HE_NORMAL",
            (weights) -> {
                double factor = Math.sqrt(2.0 / weights.length);
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        weights[i][j] = random.nextGaussian() * factor;
                    }
                }
                return weights;
            }
    );
}
