package edu.akatarin;

public class Number {
    private final double[] pixels;
    private final int value;

    public Number(double[] pixels, int value) {
        this.pixels = pixels;
        this.value = value;
    }

    public double[] getPixels() {
        return pixels;
    }

    public int getValue() {
        return value;
    }
}
