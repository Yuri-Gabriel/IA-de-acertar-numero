package com.example;

public class RedeNeural {
    public int nEntradas;
    public int[] camadasOcultas;
    public int nSaidas;

    public double[][][] pesos;
    public double[][][] bias;

    public RedeNeural(DataSet data) {
        this.nEntradas = data.images.get(0).length * data.images.get(0)[0].length;

        /**
         * Camada oculta 01: 512 neuronios
         * Camada oculta 02: 256 neuronios
         * Camada oculta 03: 128 neuronios
         * Camada oculta 04: 64 neuronios
        */
        this.camadasOcultas = new int[] {512, 256, 128, 64};

        this.nSaidas = 10;

        this.inicializarPesos();
        this.inicializarBias();

    }

    /**
     * Põe valores aleatórios nnas bias da rede
    */
    private void inicializarBias() {
        this.bias = new double[this.camadasOcultas.length + 1][][];

        for (int i = 0; i < this.camadasOcultas.length; i++) {
            int camadaAtual = this.camadasOcultas[i];
            this.bias[i] = new double[camadaAtual][1];
        }

        this.bias[this.camadasOcultas.length] = new double[this.nSaidas][1];

        for (int i = 0; i < this.bias.length; i++) {
            for (int j = 0; j < this.bias[i].length; j++) {
                this.bias[i][j][0] = Math.random() - 0.5;
            }
        }
    }

    /**
     * Põe valores aleatórios nos pesos da rede
    */
    private void inicializarPesos() {
        this.pesos = new double[this.camadasOcultas.length + 1][][];

        int camadaAnterior = this.nEntradas;

        for (int i = 0; i < this.camadasOcultas.length; i++) {
            int camadaAtual = this.camadasOcultas[i];
            this.pesos[i] = new double[camadaAtual][camadaAnterior];
            camadaAnterior = camadaAtual;
        }

        this.pesos[this.pesos.length - 1] = new double[this.nSaidas][camadaAnterior];

        for (int i = 0; i < this.pesos.length; i++) {
            for (int j = 0; j < this.pesos[i].length; j++) {
                for (int k = 0; k < this.pesos[i][j].length; k++) {
                    this.pesos[i][j][k] = Math.random() - 0.5;
                }
            }
        }
    }

    public void info() {
        System.out.println("Entradas: " + nEntradas);
        System.out.print("Camadas intermediárias: ");
        for (int c : camadasOcultas)
            System.out.print(c + " ");
        System.out.println("\nSaídas: " + nSaidas);
    }
}
