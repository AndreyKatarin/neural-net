package edu.akatarin;

import java.util.function.BiFunction;
import java.util.function.UnaryOperator;
import java.util.stream.DoubleStream;

import static java.lang.Math.exp;
import static java.lang.Math.log;

public class Activation {
    private final String name;
    private final UnaryOperator<double[]> function;
    private final BiFunction<double[], double[], double[]> derivative;

    private Activation(String name, UnaryOperator<double[]> function, BiFunction<double[], double[], double[]> derivative) {
        this.name = name;
        this.function = function;
        this.derivative = derivative;
    }

    public String getName() {
        return name;
    }

    public double[] apply(double[] x) {
        return function.apply(x);
    }

    public double[] applyDerivative(double[] x, double[] y) {
        return derivative.apply(x, y);
    }

    public static Activation ReLU = new Activation(
            "ReLU",
            x -> DoubleStream.of(x).map(v -> v <= 0 ? 0 : v).toArray(),// fn
            (x, y) -> {
                double[] dCdI = new double[x.length];
                double[] dOdI = DoubleStream.of(x).map(v -> v <= 0 ? 0 : 1).toArray();// dFn
                for (int i = 0; i < x.length; i++) {
                    dCdI[i] = dOdI[i] * y[i];
                }
                return dCdI;
            }

    );
    public static Activation Leaky_ReLU = new Activation(
            "Leaky_ReLU",
            x -> DoubleStream.of(x).map(v -> v > 0 ? v : 0.01 * v).toArray(),// fn,
            (x, y) -> {
                double[] dCdI = new double[x.length];
                double[] dOdI = DoubleStream.of(x).map(v -> v < 0 ? 0.01 : 1).toArray();// dFn
                for (int i = 0; i < x.length; i++) {
                    dCdI[i] = dOdI[i] * y[i];
                }
                return dCdI;
            }
    );
    public static Activation Sigmoid = new Activation(
            "Sigmoid",
            x -> DoubleStream.of(x).map(v -> 1.0 / (1.0 + exp(-v))).toArray(),// fn
            (x, y) -> {
                double[] dCdI = new double[x.length];
                double[] dOdI = DoubleStream.of(x).map(v -> v * (1.0 - v)).toArray();// dFn
                for (int i = 0; i < x.length; i++) {
                    dCdI[i] = dOdI[i] * y[i];
                }
                return dCdI;
            }

    );
    public static Activation Softplus = new Activation(
            "Softplus",
            x -> DoubleStream.of(x).map(v -> log(1.0 + exp(v))).toArray(),// fn
            (x, y) -> {
                double[] dCdI = new double[x.length];
                double[] dOdI = DoubleStream.of(x).map(v -> 1.0 / (1.0 + exp(-v))).toArray();// dFn
                for (int i = 0; i < x.length; i++) {
                    dCdI[i] = dOdI[i] * y[i];
                }
                return dCdI;
            }

    );
    public static Activation Identity = new Activation(
            "Identity",
            x -> x,                   // fn
            (x, y) -> {
                double[] dCdI = new double[x.length];
                double[] dOdI = DoubleStream.of(x).map(v -> 1.0).toArray();// dFn
                for (int i = 0; i < x.length; i++) {
                    dCdI[i] = dOdI[i] * y[i];
                }
                return dCdI;
            }

    );
    //Выгнутая тождественная функция
    public static Activation BentIdentity = new Activation(
            "BentIdentity",
            x -> DoubleStream.of(x)
                    .map(v -> ((Math.sqrt(Math.pow(v, 2) + 1) - 1) / 2) + v)
                    .toArray(),// fn
            (x, y) -> {
                double[] dCdI = new double[x.length];
                double[] dOdI =  DoubleStream.of(x)
                        .map(v -> 1 + (v / (2 * Math.sqrt(Math.pow(v, 2) + 1))))// dFn
                        .toArray();
                for (int i = 0; i < x.length; i++) {
                    dCdI[i] = dOdI[i] * y[i];
                }
                return dCdI;
            }
    );

    public static Activation Softmax = new Activation(
            "Softmax",
            x -> {
                double max = DoubleStream.of(x).max().orElse(0.0);
                double sum = DoubleStream.of(x).map(v -> Math.exp(v - max)).sum();
                return DoubleStream.of(x).map(v -> Math.exp(v - max) / sum).toArray();
            },
            (x, y) -> {
                double[] dCdI = new double[x.length];
                //преобразуем вектор в диагональную матрицу размерности x^2
                double[][] jacob = new double[x.length][x.length];
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x.length; j++) {
                        int kroneckerDelta = i == j ? 1 : 0;
                        jacob[i][j] = x[i] * (kroneckerDelta - x[j]);
                    }
                }
                //умножаем матрицу на вектор градиента функции ошибки
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x.length; j++) {
                        dCdI[i] += jacob[i][j] * y[j];
                    }
                }
                return dCdI;
            }
    );
}
