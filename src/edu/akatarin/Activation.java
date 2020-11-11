package edu.akatarin;

import java.util.function.UnaryOperator;
import java.util.stream.DoubleStream;

import static java.lang.Math.exp;
import static java.lang.Math.log;

public class Activation {
    private final String name;
    private final UnaryOperator<double[]> function;
    private final UnaryOperator<double[]> derivative;

    private Activation(String name, UnaryOperator<double[]> function, UnaryOperator<double[]> derivative) {
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

    public double[] applyDerivative(double[] x) {
        return derivative.apply(x);
    }

    public static Activation ReLU = new Activation(
            "ReLU",
            x -> DoubleStream.of(x).map(v -> v <= 0 ? 0 : v).toArray(),// fn
            x -> DoubleStream.of(x).map(v -> v <= 0 ? 0 : 1).toArray()// dFn

    );
    public static Activation Leaky_ReLU = new Activation(
            "Leaky_ReLU",
            x -> DoubleStream.of(x).map(v -> v > 0 ? v : 0.01 * v).toArray(),// fn,
            x -> DoubleStream.of(x).map(v -> v < 0 ? 0.01 : 1).toArray()// dFn
    );
    public static Activation Sigmoid = new Activation(
            "Sigmoid",
            x -> DoubleStream.of(x).map(v -> 1.0 / (1.0 + exp(-v))).toArray(),// fn
            x -> DoubleStream.of(x).map(v -> v * (1.0 - v)).toArray()// dFn

    );
    public static Activation Softplus = new Activation(
            "Softplus",
            x -> DoubleStream.of(x).map(v -> log(1.0 + exp(v))).toArray(),// fn
            x -> DoubleStream.of(x).map(v -> 1.0 / (1.0 + exp(-v))).toArray()// dFn

    );
    public static Activation Identity = new Activation(
            "Identity",
            x -> x,                   // fn
            x -> DoubleStream.of(x).map(v -> 1.0).toArray()// dFn

    );
    //Выгнутая тождественная функция
    public static Activation BentIdentity = new Activation(
            "BentIdentity",
            x -> DoubleStream.of(x)
                    .map(v -> ((Math.sqrt(Math.pow(v, 2) + 1) - 1) / 2) + v)
                    .toArray(),// fn
            x -> DoubleStream.of(x)
                    .map(v -> 1 + (v / (2 * Math.sqrt(Math.pow(v, 2) + 1))))
                    .toArray()// dFn
    );

    public static Activation Softmax = new Activation(
            "Softmax",
            x -> {
                double max = DoubleStream.of(x).max().orElse(0.0);
                double sum = DoubleStream.of(x).map(v -> Math.exp(v - max)).sum();
                return DoubleStream.of(x).map(v -> Math.exp(v - max) / sum).toArray();
            },
            x -> DoubleStream.of(x).map(v -> v * (1 - v)).toArray()// dFn
    );
}
