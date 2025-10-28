package com.example;

import java.io.*;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        try {
            String inputPath = System.getProperty("user.dir") + "/src/main/resources/input";

            String trainingImagesFilepath = inputPath + "/train-images-idx3-ubyte/train-images.idx3-ubyte";
            String trainingLabelsFilepath = inputPath + "/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
            String testImagesFilepath = inputPath + "/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
            String testLabelsFilepath = inputPath + "/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";

            Loader loader = new Loader();

            DataSet trainData = loader.getData(trainingImagesFilepath, trainingLabelsFilepath);
            DataSet testdata = loader.getData(testImagesFilepath, testLabelsFilepath);

            RedeNeural redeNeural = new RedeNeural(trainData);

            Treinamento treinamento = new Treinamento(redeNeural, trainData);
            treinamento.start();

        } catch (IOException e) {
            System.err.println("Erro ao carregar dados MNIST: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Erro no treinamento: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void printImage(double[][] image) {
        System.out.println("Visualização ASCII (28x28):");
        for (double[] row : image) {
            for (double pixel : row) {
                if (pixel < 0.2) System.out.print(" ");
                else if (pixel < 0.4) System.out.print(".");
                else if (pixel < 0.6) System.out.print(":");
                else if (pixel < 0.8) System.out.print("+");
                else System.out.print("#");
            }
            System.out.println();
        }
    }
}