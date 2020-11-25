package edu.akatarin;

import java.util.function.BinaryOperator;
import java.util.function.ToDoubleBiFunction;

public class CostFunction {
    private final String name;
    private final ToDoubleBiFunction<double[], double[]> function;
    private final BinaryOperator<double[]> derivative;

    private CostFunction(String name, ToDoubleBiFunction<double[], double[]> function, BinaryOperator<double[]> derivative) {
        this.name = name;
        this.function = function;
        this.derivative = derivative;
    }

    public String getName() {
        return name;
    }

    public double apply(double[] expected, double[] actual) {
        return function.applyAsDouble(expected, actual);
    }

    public double[] applyDerivative(double[] expected, double[] actual) {
        return derivative.apply(expected, actual);
    }

    // Cost function: Mean square error, C = 1/n * ∑(exp - act)^2
    public static CostFunction MSE = new CostFunction(
            "MSE",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += Math.pow(expected[i] - actual[i], 2);
                }
                return (1.0 / expected.length) * temp;
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
                for (int i = 0; i < expected.length; i++) {
                    result[i] = (2.0 / expected.length) * (expected[i] - actual[i]);
                }
                return result;
            }
    );
    //Cost function: Quadratic, C = ∑(exp−act)^2
    public static CostFunction QUADRATIC = new CostFunction(
            "QUADRATIC",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += Math.pow(expected[i] - actual[i], 2);
                }
                return temp;
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
                for (int i = 0; i < expected.length; i++) {
                    result[i] = (expected[i] - actual[i]) * 2.0;
                }
                return result;
            }
    );
    //Cost function: HalfQuadratic, C = 0.5 ∑(exp-act)^2
    public static CostFunction HALF_QUADRATIC = new CostFunction(
            "HalfQuadratic",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += Math.pow(expected[i] - actual[i], 2);
                }
                return temp * 0.5;
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
                for (int i = 0; i < expected.length; i++) {
                    result[i] = (expected[i] - actual[i]);
                }
                return result;
            }
    );
    //Cost function: CrossEntropy, C = -∑(exp*log(act))
    public static CostFunction CROSS_ENTROPY = new CostFunction(
            "CrossEntropy",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += actual[i] != 0 ? -expected[i]*Math.log(actual[i]) : 0;
                }
                return temp;
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
                for (int i = 0; i < expected.length; i++) {
                    result[i] = actual[i] != 0 ? expected[i] / actual[i] : 0; //(expected[i] - actual[i]);
                }
                return result;
            }
    );
}
