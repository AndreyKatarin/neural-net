package edu.akatarin;

import java.util.function.DoubleUnaryOperator;

public abstract class Optimizer {
    String name;
    double learningRate;
    DoubleUnaryOperator function;

    abstract String getName();
    abstract double apply(double x);


    public static class GradientDescent extends Optimizer {

        public GradientDescent(double learningRate) {
            this.name = "Gradient Descent";
            this.learningRate = learningRate;
        }

        @Override
        double apply(double x) {
            return learningRate * x;
        }

        @Override
        public String getName() {
           return name;
        }
    }

    public static class Momentum extends Optimizer {
        private final double momentum;

        public Momentum(double learningRate, double momentum) {
            this.name = "Momentum";
            this.learningRate = learningRate;
            this.momentum = momentum;
        }

        @Override
        String getName() {
            return name;
        }

        @Override
        double apply(double x) {
            return learningRate * x;
        }

        double applyMomentum( double y){
           return momentum * y;
        }
    }
}
