package edu.akatarin.util;

import edu.akatarin.Number;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Loads MNIST data sets. Loading works according to specified standards at
 * http://yann.lecun.com/exdb/mnist/.
 *
 */
public class MNISTLoader {

    /**
     * Tries to read MNIST data from some file.
     *
     * @param filePrefix filename e.g. "train-images-idx3-ubyte.gz"
     * @return imported data
     */
    public static List<Number> importData(String filePrefix) {
        List<Number> numbers = null;
        String imgFileName = filePrefix + "-images-idx3-ubyte.gz";
        String lblFileName = filePrefix + "-labels-idx1-ubyte.gz";
        byte[] buffer = new byte[4];
        try {
            System.out.println("---Importing MNIST data---");
            GZIPInputStream images = new GZIPInputStream(new FileInputStream(imgFileName));
            GZIPInputStream labels = new GZIPInputStream(new FileInputStream(lblFileName));

            images.read(buffer);
            int imagesMagicNum = bytesToInt(buffer);
            labels.read(buffer);
            int labelsMagicNum = bytesToInt(buffer);
            if (imagesMagicNum == 2051 || labelsMagicNum == 2049) {

                images.read(buffer);
                int itemCount = bytesToInt(buffer);
                labels.read(buffer);
                int labelCount = bytesToInt(buffer);

                if (itemCount == labelCount) {
                    images.read(buffer);
                    int rowCount = bytesToInt(buffer);
                    images.read(buffer);
                    int colCount = bytesToInt(buffer);

                    numbers = new ArrayList<>(itemCount);

                    int pixelCount = rowCount * colCount;
                    for (int i = 0; i < itemCount; i++) {
                        double[] image = new double[pixelCount];
                        double[] labelAsArray = new double[10];
                        int label = labels.read();
                        labelAsArray[label] = 1.0;
                        for (int j = 0; j < pixelCount; j++) {
                            image[j] = images.read() / 255.0;
                        }
                        numbers.add(new Number(image, label));
                    }
                }
            }
            System.out.println("---Finished---");
        } catch (IOException ex) {
            System.err.println("Error while reading MNIST dataset");
            ex.printStackTrace();
        }
        return numbers;
    }

    /**
     * Converts 4 bytes to 32 bit integer.
     *
     * @param bytes byte array
     * @return integer value representation
     */
    private static int bytesToInt(byte[] bytes) {
        return ((bytes[0] & 0xFF) << 24 | (bytes[1] & 0xFF) << 16 | (bytes[2] & 0xFF) << 8 | (bytes[3] & 0xFF));
    }

}