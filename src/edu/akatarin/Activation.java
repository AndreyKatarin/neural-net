package edu.akatarin;

import java.util.function.DoubleUnaryOperator;

import static java.lang.Math.exp;
import static java.lang.Math.log;

public class Activation {
    private final String name;
    private final DoubleUnaryOperator function;
    private final DoubleUnaryOperator derivative;

    private Activation(String name, DoubleUnaryOperator function, DoubleUnaryOperator derivative) {
        this.name = name;
        this.function = function;
        this.derivative = derivative;
    }

    public String getName() {
        return name;
    }

    public double apply(double x) {
        return function.applyAsDouble(x);
    }

    public double applyDerivative(double x) {
        return derivative.applyAsDouble(x);
    }

    // --------------------------------------------------------------------------
    // --- A few predefined ones ------------------------------------------------
    // --------------------------------------------------------------------------
    // The simple properties of most activation functions as stated above makes
    // it easy to create the majority of them by just providing lambdas for
    // fn and the diff dfn.
    public static Activation ReLU = new Activation(
            "ReLU",
            x -> x <= 0 ? 0 : x,                // fn
            x -> x <= 0 ? 0 : 1                 // dFn
    );
    public static Activation Leaky_ReLU = new Activation(
            "Leaky_ReLU",
            x -> x <= 0 ? 0.01 * x : x,         // fn
            x -> x <= 0 ? 0.01 : 1              // dFn
    );
    public static Activation Sigmoid = new Activation(
            "Sigmoid",
            x -> 1.0 / (1.0 + exp(-x)),         // fn
            x -> x * (1.0 - x)                  // dFn
    );
    public static Activation Softplus = new Activation(
            "Softplus",
            x -> log(1.0 + exp(x)),             // fn
            x -> 1.0 / (1.0 + exp(-x))          // dFn
    );
    public static Activation Identity = new Activation(
            "Identity",
            x -> x,                             // fn
            x -> 1                              // dFn
    );
}
