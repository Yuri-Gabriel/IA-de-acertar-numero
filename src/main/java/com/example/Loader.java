package com.example;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Loader {
    private DataSet readImagesLabels(String imagesFilepath, String labelsFilepath) throws IOException {
        double[][] labels = null;

        try (DataInputStream labelFile = new DataInputStream(
                new BufferedInputStream(new FileInputStream(labelsFilepath)))) {
            int magic = labelFile.readInt();
            if (magic != 2049) {
                throw new IllegalArgumentException(
                    "Magic number mismatch, expected 2049, got " + magic);
            }
            int size = labelFile.readInt();
            labels = new double[size][10];
            for (int i = 0; i < size; i++) {
                int value = labelFile.readUnsignedByte();
                labels[i][value] = 1.0;
            }
        }

        List<double[][]> images = new ArrayList<>();
        try (DataInputStream imageFile = new DataInputStream(
                new BufferedInputStream(new FileInputStream(imagesFilepath)))) {
            int magic = imageFile.readInt();
            if (magic != 2051) {
                throw new IllegalArgumentException(
                    "Magic number mismatch, expected 2051, got " + magic);
            }
            int size = imageFile.readInt();
            int rows = imageFile.readInt();
            int cols = imageFile.readInt();

            for (int i = 0; i < size; i++) {
                double[][] image = new double[rows][cols];
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        image[r][c] = imageFile.readUnsignedByte() / 255.0;
                    }
                }
                images.add(image);
            }
        }

        return new DataSet(images, labels);
    }


    public DataSet getData(
        String imagesFilepath, 
        String labelsFilepath
    ) throws IOException {
        return readImagesLabels(imagesFilepath, labelsFilepath);
    }
}
