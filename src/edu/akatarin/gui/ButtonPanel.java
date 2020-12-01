package edu.akatarin.gui;

import edu.akatarin.Number;
import edu.akatarin.util.MNISTLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Panel holding all the buttons and interacting with neural network.
 *
 * @author ChriZ98
 */
public class ButtonPanel extends JPanel {

    private final JLabel[] labels;
    private final Frame frame;
    private final JButton predictButton;

    /**
     * Initializes all buttons and labels.
     *
     * @param frame the parent frame
     */
    public ButtonPanel(Frame frame) {
        super(new GridLayout(12, 3));
        this.frame = frame;
        this.labels = new JLabel[10];

        for (int i = 0; i < 10; i++) {
            add(new JLabel(i + " ", JLabel.RIGHT));
            labels[i] = new JLabel("", JLabel.CENTER);
            add(labels[i]);
            add(new JLabel("%"));
        }
        JButton resetButton = new JButton("Reset");
        JButton trainButton = new JButton("Train");
        JButton saveButton = new JButton("Save");
        JButton loadButton = new JButton("Load");
        JButton testButton = new JButton("Test");
        predictButton = new JButton("Predict");

        resetButton.addActionListener((ActionEvent ae) -> {
            resetButtonActionPerformed();
        });
        trainButton.addActionListener((ActionEvent ae) -> {
            trainButtonActionPerformed();
        });
        saveButton.addActionListener((ActionEvent ae) -> {
            saveButtonActionPerformed();
        });
        loadButton.addActionListener((ActionEvent ae) -> {
            loadButtonActionPerformed();
        });
        testButton.addActionListener((ActionEvent ae) -> {
            testButtonActionPerformed();
        });
        predictButton.addActionListener((ActionEvent ae) -> {
            predictButtonActionPerformed();
        });

        add(loadButton);
        add(saveButton);
        add(resetButton);
        add(trainButton);
        add(testButton);
        add(predictButton);

        Graphics2D g = frame.getGraphics2D();
        g.setBackground(Color.WHITE);
        g.clearRect(0, 0, Frame.DRAW_SIZE, Frame.DRAW_SIZE);
        setLabels(new double[10]);
        frame.repaint();
    }

    /**
     * Trains the network using MNIST data.
     */
    private void trainButtonActionPerformed() {
        List<Number> numbers = MNISTLoader.importData("data/train");
        int batchSize = 10;
        int epoch = 0;
        int batchCount = numbers.size() / batchSize;
        double[][] expectedOutputs = new double[batchSize][10];
        double batchError;
        do {
            Collections.shuffle(numbers);
            epoch++;
            for (int i = 0; i < batchCount; i++) {
                int fromIndex = i * batchSize;
                int toIndex = Math.min(numbers.size(), (i + 1) * batchSize);
                List<Number> batch = numbers.subList(fromIndex, toIndex);
                for (int j = 0; j < batchSize; j++) {
                    expectedOutputs[j] = batch.get(j).getIdealOut();
                }
                double totalBatchError = frame.getNet().trainBatch(batch.toArray(new Number[batchSize]), expectedOutputs);
                batchError = totalBatchError / batchSize;
                System.out.print("\rEpoch: " + epoch + " Batch: " + i + " of " + batchCount + " ERROR: " + totalBatchError + " AVG ERR: " + batchError);
            }
        } while ( epoch <= 3);
    }

    /**
     * Testing the network by iterating through some test images and displaying
     * calculated results.
     */
    private void testButtonActionPerformed() {
        Random rand = new Random();
        List<Number> testNumbers = MNISTLoader.importData("data/t10k");
        new Thread(() -> {
            System.out.println("\n---Testing Network---");
            for (int j = 0; j < testNumbers.size(); j += rand.nextInt(200) + 400) {
                try {
                    drawImageFromVector(testNumbers.get(j).getPixels());
                    Thread.sleep(1000);
                } catch (InterruptedException ex) {
                }
            }
            System.out.println("finished");
            frame.setPredicted(true);
        }).start();
    }

    /**
     * Saves the current neural network.
     */
    private void saveButtonActionPerformed() {
        //NetworkIO.saveNetwork("data/network.dat", frame.getNet());
    }

    /**
     * Loads some saved neural network.
     */
    private void loadButtonActionPerformed() {
        //frame.setNet(NetworkIO.loadNetwork("data/network.dat"));
    }

    /**
     * Resets the drawing panel and the labels.
     */
    public void resetButtonActionPerformed() {
        Graphics2D g = frame.getGraphics2D();
        g.setBackground(Color.WHITE);
        g.clearRect(0, 0, Frame.DRAW_SIZE, Frame.DRAW_SIZE);
        setLabels(new double[10]);
        frame.setPredicted(false);
        frame.repaint();
    }

    /**
     * Sets the probabilities as label content.
     *
     * @param output calculated probabilities
     */
    public void setLabels(double[] output) {
        double sum = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (double d : output) {
            sum += d;
            if (d > max) {
                max = d;
            }
        }
        for (int i = 0; i < 10; i++) {
            if (sum != 0) {
                labels[i].setText(String.format("%.2f", output[i] / sum * 100));
            } else {
                labels[i].setText(String.format("%.2f", output[i] * 100));
            }
            if (output[i] == max && sum != 0) {
                labels[i].setForeground(new Color(0f, 0.8f, 0f));
            } else {
                labels[i].setForeground(Color.BLACK);
            }
        }
    }

    /**
     * Generates a double array representing the drawing. White pixels are
     * encoded as 0. Black pixels are encoded as 1. Grey pixels are values
     * inbetween.
     *
     * @param image drawing to represent
     * @return double array with image data
     */
    public double[] calcInput(BufferedImage image) {
        double[] input = new double[784];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                input[i * 28 + j] = 1.0 - Math.abs((image.getRGB(j, i) + 1.0) / 16777215.0);
            }
        }
        return input;
    }

    /**
     * Draws image from vector. The vector contains values between 0 and 1. That
     * is converted into different grey colors and then painted out.
     *
     * @param vector vector to be shown on panel
     */
    private void drawImageFromVector(double[] vector) {
        resetButtonActionPerformed();
        Graphics2D g = frame.getGraphics2D();
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int grey = 255 - (int) (vector[i * 28 + j] * 255);
                g.setColor(new Color(grey, grey, grey, 255));
                g.fillRect(j * 25, i * 25, 25, 25);
            }
        }
        frame.repaint();
        frame.getNet().feedForward(vector);
        setLabels(frame.getNet().getOutput());
    }

    /**
     * Counts rows of white pixels in current drawing.
     *
     * @param image image to analyze
     * @param i1 start index
     * @param i2 direchtion. either -1 or 1
     * @param horizontal count rows if true. count cols if false.
     * @return number of rows with white pixels
     */
    private int getWhitespaceInImage(BufferedImage image, int i1, int i2, boolean horizontal) {
        int cut = i1;
        for (int i = i1; i >= 0 && i < Frame.DRAW_SIZE; i += i2) {
            boolean white = true;
            for (int j = 0; j < Frame.DRAW_SIZE; j++) {
                if ((horizontal && new Color(image.getRGB(j, i)).getRed() != 255)
                        || (!horizontal && new Color(image.getRGB(i, j)).getRed() != 255)) {
                    white = false;
                    break;
                }
            }

            if (white || i2 < 0) {
                cut = i + 1;
            }
            if (!white) {
                break;
            }
        }
        return cut;
    }

    /**
     * Calculates the center of an image with the highest pixel density.
     *
     * @param image image to analyze
     * @return vector containing y and x value of center
     */
    private double[] centerOfMassOfPixels(BufferedImage image) {
        double[] yxVector = new double[2];
        int iCount = 0, jCount = 0;
        int iVal = 0, jVal = 0;
        for (int i = 0; i < image.getHeight(); i++) {
            for (int j = 0; j < image.getWidth(); j++) {
                if (new Color(image.getRGB(j, i)).getRed() != 255) {
                    iVal += i;
                    jVal += j;
                    iCount++;
                    jCount++;
                }
            }
        }
        yxVector[0] = iVal / iCount;
        yxVector[1] = jVal / jCount;
        return yxVector;
    }

    /**
     * Translates the image towards some point.
     *
     * @param image image to translate
     * @param iC new center y
     * @param jC new center x
     * @param iM image center y
     * @param jM image center x
     */
    private void moveTowardsPoint(BufferedImage image, int iC, int jC, int iM, int jM) {
        int iDiff = iM - iC;
        int jDiff = jM - jC;

        if (iDiff > 0) {
            for (int i = image.getHeight() - 1; i >= 0; i--) {
                for (int j = 0; j < image.getWidth(); j++) {
                    if (i - iDiff >= 0) {
                        image.setRGB(j, i, image.getRGB(j, i - iDiff));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        } else if (iDiff < 0) {
            for (int i = 0; i < image.getHeight(); i++) {
                for (int j = 0; j < image.getWidth(); j++) {
                    if (i - iDiff < image.getHeight()) {
                        image.setRGB(j, i, image.getRGB(j, i - iDiff));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        }
        if (jDiff > 0) {
            for (int i = 0; i < image.getHeight(); i++) {
                for (int j = image.getWidth() - 1; j >= 0; j--) {
                    if (j - jDiff >= 0) {
                        image.setRGB(j, i, image.getRGB(j - jDiff, i));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        } else if (jDiff < 0) {
            for (int i = 0; i < image.getHeight(); i++) {
                for (int j = 0; j < image.getWidth(); j++) {
                    if (j - jDiff < image.getWidth()) {
                        image.setRGB(j, i, image.getRGB(j - jDiff, i));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        }
    }

    /**
     * Predicts the digit drawn by using the neural network to calculate.
     */
    public void predictButtonActionPerformed() {
        if (frame.isPredicted()) {
            return;
        }
        Thread t1 = new Thread(() -> {
            try {
                int nCut = getWhitespaceInImage(frame.getImage(), 0, 1, true);
                int wCut = getWhitespaceInImage(frame.getImage(), 0, 1, false);
                int eCut = getWhitespaceInImage(frame.getImage(), Frame.DRAW_SIZE - 1, -1, false);
                int sCut = getWhitespaceInImage(frame.getImage(), Frame.DRAW_SIZE - 1, -1, true);

                int width = eCut - wCut;
                int height = sCut - nCut;

                if (width <= 0 || height <= 0) {
                    return;
                }

                BufferedImage cutImage = new BufferedImage(eCut - wCut, sCut - nCut, BufferedImage.TYPE_INT_ARGB_PRE);
                ((Graphics2D) cutImage.getGraphics()).drawImage(frame.getImage().getSubimage(wCut, nCut, width, height), 0, 0, null);

                resetButtonActionPerformed();

                int offset = (int) (0.14285714285714285714 * Frame.DRAW_SIZE);
                if (width == height) {
                    frame.getGraphics2D().drawImage(cutImage, offset, offset, Frame.DRAW_SIZE - offset, Frame.DRAW_SIZE - offset, 0, 0, width, height, this);
                } else if (width > height) {
                    double scaledWidth = (Frame.DRAW_SIZE - (2.0 * offset)) / width;
                    int space = (int) ((width - height) * scaledWidth / 2.0);
                    frame.getGraphics2D().drawImage(cutImage, offset, offset + space, Frame.DRAW_SIZE - offset, Frame.DRAW_SIZE - offset - space, 0, 0, width, height, this);
                } else {
                    double scaledHeight = (Frame.DRAW_SIZE - (2.0 * offset)) / height;
                    int space = (int) ((height - width) * scaledHeight / 2.0);
                    frame.getGraphics2D().drawImage(cutImage, offset + space, offset, Frame.DRAW_SIZE - offset - space, Frame.DRAW_SIZE - offset, 0, 0, width, height, this);
                }
                frame.repaint();
                Thread.sleep(400);

                double[] cop = centerOfMassOfPixels(frame.getImage());

                moveTowardsPoint(frame.getImage(), (int) cop[0], (int) cop[1], Frame.DRAW_SIZE / 2, Frame.DRAW_SIZE / 2);
                frame.repaint();
                Thread.sleep(400);

                double[] vec = new double[784];
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        int sum = 0;
                        for (int k = 0; k < 25; k++) {
                            for (int l = 0; l < 25; l++) {
                                sum += new Color(frame.getImage().getRGB(j * 25 + k, i * 25 + l)).getRed();
                            }
                        }
                        vec[i * 28 + j] = 1 - sum / 625.0 / 255.0;
                    }
                }
                drawImageFromVector(vec);
                frame.setPredicted(true);
            } catch (InterruptedException ex) {
            }
        });
        t1.start();
    }

    /**
     * Gets the predict button.
     *
     * @return predict button
     */
    public JButton getPredictButton() {
        return predictButton;
    }
}