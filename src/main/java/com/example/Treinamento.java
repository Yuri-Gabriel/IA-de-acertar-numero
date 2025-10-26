package com.example;

import java.util.ArrayList;
import java.util.List;

public class Treinamento {
    private RedeNeural redeNeural;
    private DataSet dataSet;
    private double tx_aprendizado;
 
    public Treinamento(RedeNeural redeNeural, DataSet dataSet) {
        this.redeNeural = redeNeural;
        this.dataSet = dataSet;
        this.tx_aprendizado = 0.1;
    }

    public void start() throws Exception {
        for(int i = 0; i < this.dataSet.images.size(); i++) {
            double[][] imagemEntrada = this.dataSet.images.get(i);
            Main.printImage(imagemEntrada);
            double[][] flatInput = this.flatten(imagemEntrada);

            List<double[][]>[] forwardResult = this.forward(
                flatInput
            );

            System.out.println("========================================");
            System.out.println("--- Saída da Camada (10 neurônios) ---");
            
            List<double[][]> ativacoes = forwardResult[1];
            
            double[][] saidaRede = ativacoes.get(ativacoes.size() - 1);

            double maxAtivacao = -1;
            int neuronioPrevisto = -1;

            for(int n = 0; n < saidaRede.length; n++) {
                System.out.printf("Neurônio %d: %.4f\n", n, saidaRede[n][0]);
                if (saidaRede[n][0] > maxAtivacao) {
                    maxAtivacao = saidaRede[n][0];
                    neuronioPrevisto = n;
                }
            }

            System.out.println("----------------------------------------");
            
            System.out.println("Previsão da Rede (maior ativação): " + neuronioPrevisto);
            int esperado = -1;
            double[][] label = new double[this.dataSet.labels[i].length][1];
            for (int j = 0; j < this.dataSet.labels[i].length; j++) {
                label[j][0] = this.dataSet.labels[i][j];
                if(label[j][0] == 1.0) {
                    esperado = j;
                }
            }
            System.out.print("Valor esperado: " + esperado + "\n");

            List<double[][]>[] backPropagationResult = this.backPropagation(
                forwardResult[0], forwardResult[1], label, flatInput
            );

            this.recalcularPesos(backPropagationResult[0], backPropagationResult[1]);
        }
        
    }

    private List<double[][]>[] forward(double[][] matrizA) throws Exception {

        List<double[][]> z = new ArrayList<>();
        List<double[][]> a = new ArrayList<>();

        for (int i = 0; i < this.redeNeural.camadasOcultas.length; i++) {
            double[][] pesos = this.redeNeural.pesos[i];
            double[][] bias = this.redeNeural.bias[i];

            double[][] zAtual = this.sumMatriz(
                this.multMatriz(pesos, matrizA),
                bias
            );
            z.add(zAtual);
            double[][] aAtual = this.ativarValores(zAtual);
            a.add(aAtual);
            matrizA = aAtual;
        }

        int ultima = this.redeNeural.camadasOcultas.length;
        double[][] pesosSaida = this.redeNeural.pesos[ultima];
        double[][] biasSaida = this.redeNeural.bias[ultima];

        double[][] zSaida = this.sumMatriz(
            this.multMatriz(pesosSaida, matrizA),
            biasSaida
        );
        z.add(zSaida);

        double[][] aSaida = this.ativarValores(zSaida);
        a.add(aSaida);

        List<double[][]>[] resultado = (List<double[][]>[]) new List[2];
        resultado[0] = z;
        resultado[1] = a;

        return resultado;
    }

    private double[][] flatten(double[][] matriz) {
        int rows = matriz.length;
        int cols = matriz[0].length;
        double[][] vetor = new double[rows * cols][1];
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                vetor[index++][0] = matriz[i][j];
            }
        }
        return vetor;
    }


    private List<double[][]>[] backPropagation(List<double[][]> z, List<double[][]> a, double[][] label, double[][] flatInput) throws Exception {
        List<double[][]> e = new ArrayList<double[][]>();
        List<double[][]> gw = new ArrayList<double[][]>();
        List<double[][]> gb = new ArrayList<double[][]>();

        int numHiddenLayers = this.redeNeural.camadasOcultas.length;
        int outputLayerIndex = numHiddenLayers;

        for(int i = outputLayerIndex; i >= 0; i--) {
            
            if(i == outputLayerIndex) {
                e.add(
                    this.multProdPorProd(
                        this.subMatriz(
                            a.get(i),
                            label 
                        ),
                        this.desativarValores(z.get(i))
                    )
                );
                gw.add(
                    this.multMatriz(
                        e.get(0),
                        this.transporMatriz(a.get(i - 1))
                    )
                );
                gb.add(e.get(0));

            } else {
                double[][] delta = this.multProdPorProd(
                    this.multMatriz(
                        this.transporMatriz(
                            this.redeNeural.pesos[i + 1]
                        ),
                        e.get(e.size() - 1)
                    ),
                    this.desativarValores(z.get(i)) 
                );
                e.add(delta);
                
                double[][] prevActivation;
                if (i == 0) {
                    prevActivation = flatInput; 
                } else {
                    prevActivation = a.get(i - 1);
                }

                gw.add(
                    this.multMatriz(
                        e.get(e.size() - 1),
                        this.transporMatriz(prevActivation)
                    )
                );
                gb.add(e.get(e.size() - 1));
            }
        }

        gw = this.inverter(gw);
        gb = this.inverter(gb);

        List<double[][]>[] backPropagationResult = (List<double[][]>[]) new List[2];
        backPropagationResult[0] = gw;
        backPropagationResult[1] = gb;

        return backPropagationResult;
    }

    private void recalcularPesos(List<double[][]> gw, List<double[][]> gb) throws Exception {
        for(int i = 0; i < this.redeNeural.pesos.length; i++) {
            this.redeNeural.pesos[i] = this.subMatriz(
                this.redeNeural.pesos[i],
                this.multPorNum(gw.get(i), this.tx_aprendizado)
            );
            this.redeNeural.bias[i] = this.subMatriz(
                this.redeNeural.bias[i],
                this.multPorNum(gb.get(i), this.tx_aprendizado)
            );
        }
    }

    private double[][] multProdPorProd(double[][] a, double[][] b) {
        double[][] res = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a[0].length; j++)
                res[i][j] = a[i][j] * b[i][j];
        return res;
    }

    private double[][] multPorNum(double[][] matrizA, double num) {
        double[][] resultMatriz = new double[matrizA.length][matrizA[0].length];
        
        for (int i = 0; i < matrizA.length; i++) {
            for (int j = 0; j < matrizA[0].length; j++) {
                resultMatriz[i][j] = matrizA[i][j] * num;
            }
        }

        return resultMatriz;
    }

    private List<double[][]> inverter(List<double[][]> lista) {
        List<double[][]> lista_invertida = new ArrayList<double[][]>();

        for(int i = lista.size() - 1; i >= 0; i--) {
            lista_invertida.add(lista.get(i));
        }

        return lista_invertida;
    }

    private double[][] multMatriz(double[][] matrizA, double[][] matrizB) throws Exception {
        int rowsA = matrizA.length;
        int colsA = matrizA[0].length;
        int rowsB = matrizB.length;
        int colsB = matrizB[0].length;

        if (colsA != rowsB) throw new Exception(
            String.format(
                "Não é possível multiplicar as matrizes: \nmatrizA: %d x %d \nmatrizB: %d x %d ",
                matrizA.length, matrizA[0].length, matrizB.length, matrizB[0].length
            )
        );

        double[][] resultMatriz = new double[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    resultMatriz[i][j] += matrizA[i][k] * matrizB[k][j];
                }
            }
        }

        return resultMatriz;
    }

    private double[][] sumMatriz(double[][] matrizA, double[][] matrizB) throws Exception {
        if (!(matrizA.length == matrizB.length && matrizA[0].length == matrizB[0].length)) throw new Exception(
            String.format(
                "As matrizes não têm as mesmas dimensões: \nmatrizA: %d x %d \nmatrizB: %d x %d ",
                matrizA.length, matrizA[0].length, matrizB.length, matrizB[0].length
            )
        );
        
            
        int rowsA = matrizA.length;
        int colsB = matrizB[0].length;

        double[][] resultMatriz = new double[rowsA][colsB];
        
        for (int i = 0; i < matrizA.length; i++) {
            for (int j = 0; j < matrizA[0].length; j++) {
                resultMatriz[i][j] = matrizA[i][j] + matrizB[i][j];
            }
        }

        return resultMatriz;
    }

    private double[][] subMatriz(double[][] matrizA, double[][] matrizB) throws Exception {
        if (!(matrizA.length == matrizB.length && matrizA[0].length == matrizB[0].length)) throw new Exception(
            String.format(
                "As matrizes não têm as mesmas dimensões: \nmatrizA: %d x %d \nmatrizB: %d x %d ",
                matrizA.length, matrizA[0].length, matrizB.length, matrizB[0].length
            )
        );
            
        int rowsA = matrizA.length;
        int colsB = matrizB[0].length;

        double[][] resultMatriz = new double[rowsA][colsB];
        
        for (int i = 0; i < matrizA.length; i++) {
            for (int j = 0; j < matrizA[0].length; j++) {
                resultMatriz[i][j] = matrizA[i][j] - matrizB[i][j];
            }
        }

        return resultMatriz;
    }

    private double[][] transporMatriz(double[][] matriz) {
        double[][] matrizTransposta = new double[matriz[0].length][matriz.length];

        for (int i = 0; i < matriz.length; i++) {
            for (int j = 0; j < matriz[0].length; j++) {
                matrizTransposta[j][i] = matriz[i][j];
            }
        }

        return matrizTransposta;
    }

    public double[][] ativarValores(double[][] values) {
        double[][] resultMatriz = new double[values.length][values[0].length];
        
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                resultMatriz[i][j] = this.sigmoid(values[i][j]);
            }
        }

        return resultMatriz;
    }

    public double[][] desativarValores(double[][] values) {
        double[][] resultMatriz = new double[values.length][values[0].length];
        
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                resultMatriz[i][j] = this.dsigmoid(values[i][j]);
            }
        }

        return resultMatriz;
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(z * -1));
    }

    private double dsigmoid(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
}
