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

    public static void showSampleImages(DataSet trainData, DataSet testData) {
        Random random = new Random();
        
        System.out.println("\n=== AMOSTRAS DE IMAGENS DE TREINAMENTO ===");
        for (int i = 0; i < 10; i++) {
            int idx = random.nextInt(trainData.images.size());
            System.out.println("\nImagem de treinamento [" + idx + "] = " + trainData.labels[idx]);
            printImage(trainData.images.get(idx));
        }

        // System.out.println("\n=== AMOSTRAS DE IMAGENS DE TESTE ===");
        // for (int i = 0; i < 5; i++) {
        //     int idx = random.nextInt(testData.images.size());
        //     System.out.println("\nImagem de teste [" + idx + "] = " + testData.labels.get(idx));
        //     printImage(testData.images.get(idx));
        // }
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