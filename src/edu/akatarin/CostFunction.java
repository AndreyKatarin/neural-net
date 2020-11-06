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

    public double apply(double[] expected, double[] actual) {
        return function.applyAsDouble(expected, actual);
    }

    public double[] applyDerivative(double[] expected, double[] actual) {
        return derivative.apply(expected, actual);
    }

    // Cost function: Mean square error, C = 1/n * ∑(act-exp)^2
    public static CostFunction MSE = new CostFunction(
            "MSE",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += Math.pow(actual[i] - expected[i], 2);
                }
                return (1.0 / expected.length) * temp;
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
                for (int i = 0; i < expected.length; i++) {
                    result[i] = (2.0 / expected.length) * (actual[i] - expected[i]);
                }
                return result;
            }
    );
    //Cost function: Quadratic, C = ∑(y−exp)^2
    public static CostFunction QUADRATIC = new CostFunction(
            "QUADRATIC",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += Math.pow(actual[i] - expected[i], 2);
                }
                return temp;
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
                for (int i = 0; i < expected.length; i++) {
                    result[i] = (actual[i] - expected[i]) * 2.0;
                }
                return result;
            }
    );
    //Cost function: HalfQuadratic, C = 0.5 ∑(y−exp)^2
    public static CostFunction HALF_QUADRATIC = new CostFunction(
            "HalfQuadratic",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += Math.pow((actual[i] - expected[i]), 2);
                }
                return temp * 0.5;
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
                for (int i = 0; i < expected.length; i++) {
                    result[i] = (actual[i] - expected[i]);
                }
                return result;
            }
    );
    //Mean Absolute Error C = 1/n * ∑(act-exp)
    public static CostFunction MAE = new CostFunction(
            "MAE",
            (expected, actual) -> {
                double temp = 0;
                for (int i = 0; i < expected.length; i++) {
                    temp += (actual[i] - expected[i]);
                }
                return Math.abs(temp / expected.length);
            },
            (expected, actual) -> {
                double[] result = new double[expected.length];
//                for (int i = 0; i < expected.length; i++) {
//                    result[i] = (actual[i]-expected[i]);
//                }
                return result;
            }
    );
}
