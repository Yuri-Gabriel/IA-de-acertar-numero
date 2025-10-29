package com.example;

import java.util.ArrayList;
import java.util.List;

public class Treinamento {
    private RedeNeural redeNeural;
    private DataSet dataSet;
    private double tx_aprendizado;
    private double MSE;
 
    public Treinamento(RedeNeural redeNeural, DataSet dataSet) {
        this.redeNeural = redeNeural;
        this.dataSet = dataSet;
        this.tx_aprendizado = 0.1;

        this.MSE = 0.0;
    }

    public void start() throws Exception {
        for(int i = 0; i < this.dataSet.images.size(); i++) {
            double[][] imagemEntrada = this.dataSet.images.get(i);

            double[][] entrada_achatada = this.achatar(imagemEntrada);

            // Forward (ida)
            List<double[][]>[] forwardResult = this.forward(
                entrada_achatada
            );
            
            List<double[][]> ativacoes = forwardResult[1];
            
            double[][] saidaRede = ativacoes.get(ativacoes.size() - 1);
            
            /**
             * Achata o resultado esperado como foi feito com os valores de entrada
            */
            double[][] saida_esperada = new double[this.dataSet.labels[i].length][1];
            for (int j = 0; j < this.dataSet.labels[i].length; j++) {
                saida_esperada[j][0] = this.dataSet.labels[i][j];
            }

            // Backpropagation (volta)
            List<double[][]>[] backPropagationResult = this.backPropagation(
                forwardResult[0], forwardResult[1], saida_esperada, entrada_achatada
            );

            System.out.println("****************************************");
            System.out.println("\u001B[33mTREINAMENTO DA REDE\u001B[0m");
            System.out.println("****************************************");
            this.exibirDadosDaEpoca(imagemEntrada, saidaRede, saida_esperada, i + 1);
            System.out.println("****************************************");

            // Faz o rebalanceamento dos pesos
            this.recalcularPesos(backPropagationResult[0], backPropagationResult[1]);
        }
    }

    public void testar(DataSet dataSetTeste) throws Exception {
        for(int i = 0; i < dataSetTeste.images.size(); i++) {
            double[][] imagemEntrada = dataSetTeste.images.get(i);

            double[][] entrada_achatada = this.achatar(imagemEntrada);

            List<double[][]>[] forwardResult = this.forward(
                entrada_achatada
            );
            
            List<double[][]> ativacoes = forwardResult[1];
            
            double[][] saidaRede = ativacoes.get(ativacoes.size() - 1);
            
            double[][] saida_esperada = new double[dataSetTeste.labels[i].length][1];
            for (int j = 0; j < dataSetTeste.labels[i].length; j++) {
                saida_esperada[j][0] = dataSetTeste.labels[i][j];
            }

            System.out.println("****************************************");
            System.out.println("\\u001B[33mTESTANDO A REDE\\u001B[0m");
            System.out.println("****************************************");
            this.exibirDadosDaEpoca(imagemEntrada, saidaRede, saida_esperada, i + 1);
            System.out.println("****************************************");
        }
    }
    private void exibirDadosDaEpoca(double[][] imagem, double[][] saidaRede, double[][] saida_esperada, int epoca) {
        double maxAtivacao = -1;
        int neuronioPrevisto = -1;

        if(imagem != null) {
              System.out.println("Visualização ASCII (28x28):");
            for (double[] row : imagem) {
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
      

        System.out.println("========================================");
        System.out.println("Epoca: " + epoca);
        System.out.println("========================================");
        System.out.println("--- Saída da Camada (10 neurônios) ---");

        for(int n = 0; n < saidaRede.length; n++) {
            System.out.printf("Neurônio %d: %.4f\n", n, saidaRede[n][0]);
            if (saidaRede[n][0] > maxAtivacao) {
                maxAtivacao = saidaRede[n][0];
                neuronioPrevisto = n;
            }
        }

        int esperado = -1;
        for (int j = 0; j < saida_esperada.length; j++) {
            if(saida_esperada[j][0] == 1.0) {
                esperado = j;
            }
        }

        String corString = "";
        if(esperado == neuronioPrevisto) {
            corString = "\u001B[32m"; // Verde
        } else {
            corString = "\u001B[31m"; // Vermelho
        }
        
        System.out.println("----------------------------------------");
        System.out.println(corString + "Previsão da Rede (maior ativação): " + neuronioPrevisto + "\u001B[0m");
        System.out.println("Valor esperado: " + esperado);
        System.out.println("Erro Quadrático Médio (MSE): " + this.MSE);
        System.out.println("----------------------------------------");
    }

    private double calc_MSE(double[][] saidaRede, double[][] saidaEsperada) {
        double sum_erros = 0.0;
        double media = 0.0;
        for (int i = 0; i < saidaRede.length; i++) {
            sum_erros += 0.5 * Math.pow((saidaRede[i][0] - saidaEsperada[i][0]), 2);
        }
        media = sum_erros / saidaRede.length;
        return media;
    }

    private List<double[][]>[] forward(double[][] matrizA) throws Exception {

        // Lista com o resultado de cada (Wn * Xn + Bn)
        List<double[][]> z = new ArrayList<>();

        // Lista com os valores de z depois de passarem pela função de ativação 
        List<double[][]> a = new ArrayList<>();

        // Percorre paenas as camadas ocultas
        for (int i = 0; i < this.redeNeural.camadasOcultas.length; i++) {
            // Pesos entre as camadas atual x proxima
            double[][] pesos = this.redeNeural.pesos[i];

            // Bias da proxima camada
            double[][] bias = this.redeNeural.bias[i];

            /**
             * -> pesos_atuais @ entrada_atual + bias_prox_camada
            */
            double[][] zAtual = this.sumMatriz(
                this.multMatriz(pesos, matrizA),
                bias
            );
            z.add(zAtual);

            // Valores do ultimo z ativados
            double[][] aAtual = this.ativarValores(zAtual);
            a.add(aAtual);

            // Move a camada pra frente
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

        this.MSE = calc_MSE(aSaida, matrizA);

        List<double[][]>[] resultado = (List<double[][]>[]) new List[2];
        resultado[0] = z;
        resultado[1] = a;

        return resultado;
    }

    /**
     * Literalmente achata ->||<- uma matriz
     * [[1], [2], [3]],
     * [[4], [5], [6]],
     * [[7], [8], [9]]
     * 
     * [
     *  [1],
     *  [2],
     *  [3],
     *  [4],
     *  ...
     * ]
     * 
     * @param matriz - matriz a ser achatada
     * @return matriz achatada
    */
    private double[][] achatar(double[][] matriz) {
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


    /**
     * Função que realiza o BackPropagation na rede neural
     * 
     * @param z - Lista com o resultado de cada (Wn * Xn + Bn)
     * @param a - Lista com os valores de z depois de passarem pela função de ativação 
     * @param saida_esperada - Resultado esperado
     * @param flatInput - Valores de entrada da rede na epoca atual
    */
    private List<double[][]>[] backPropagation(List<double[][]> z, List<double[][]> a, double[][] saida_esperada, double[][] entrada_achatada) throws Exception {
        // Erros
        List<double[][]> e = new ArrayList<double[][]>();

        // Gradientes dos pesos
        List<double[][]> gw = new ArrayList<double[][]>();

        // Gradientes das bias
        List<double[][]> gb = new ArrayList<double[][]>();

        int numCamadasOcultas = this.redeNeural.camadasOcultas.length;

        // Varrendo as camadas de trás pra frente
        for(int i = numCamadasOcultas; i >= 0; i--) {
            /**
             * Se for a ultima camada (intermediaria 4 <- saida)
            */
            if(i == numCamadasOcultas) {
                /**
                 * -> (a[i] - matriz_esperada) * dsigmoid(z[i]
                 * OBS: A multiplicação '*' é uma multiplicação valor por valor da matriz, como se fosse uma soma entre matrizes 
                */
                e.add(
                    this.multProdPorProd(
                        this.subMatriz(
                            a.get(i),
                            saida_esperada 
                        ),
                        this.desativarValores(z.get(i))
                    )
                );
                /**
                 * -> e[0] @ a_transposta(i - 1)
                 * OBS: A multiplicação '@' é de fato uma multiplicação entre matrizes, diferente do '*' 
                */
                gw.add(
                    this.multMatriz(
                        e.get(0),
                        this.transporMatriz(a.get(i - 1))
                    )
                );
                gb.add(e.get(0));

            } else {
                /**
                 * -> matriz_delta_err = (matriz_de_pesos_transposta[i + 1] @ erros_da_camada_anterior) * dsigmoid(z[i])
                 * OBS: z[i] é o valor dos pesos (Wn * Xn + Bn) na camada i
                */
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
                
                // 
                double[][] camada_anterior_ativada;
                // Se for a primeira camada
                if (i == 0) {
                    // Recebe os valores de entrada da rede
                    camada_anterior_ativada = entrada_achatada; 
                } else {
                    // Recebe os valores dos neuronios da camada atual
                    camada_anterior_ativada = a.get(i - 1);
                }

                /**
                 * -> ultimos_erros_adicionados @ a_transposto
                 * 
                 * a_transposto = valores da camada atual ativados e transpostos
                */
                gw.add(
                    this.multMatriz(
                        e.get(e.size() - 1),
                        this.transporMatriz(camada_anterior_ativada)
                    )
                );
                // Adiciona a gb o ultimo erro da lista de erros
                gb.add(e.get(e.size() - 1));
            }
        }

        // Inverte as listas com os gradientes
        gw = this.inverter(gw);
        gb = this.inverter(gb);

        List<double[][]>[] backPropagationResult = (List<double[][]>[]) new List[2];
        backPropagationResult[0] = gw;
        backPropagationResult[1] = gb;

        return backPropagationResult;
    }

    /**
     * @param gw - Gradientes dos pesos
     * @param gb - Gradientes das bias
    */
    private void recalcularPesos(List<double[][]> gw, List<double[][]> gb) throws Exception {
        // Percorre a lista com as matrizes de pesos e bias
        for(int i = 0; i < this.redeNeural.pesos.length; i++) {
            /**
             * Rebalanceia os pesos da camada atual
             * pesos_atuais - taxa_de_aprendizado * gradientes_dos_pesos_atuais
            */
            this.redeNeural.pesos[i] = this.subMatriz(
                this.redeNeural.pesos[i],
                this.multPorNum(gw.get(i), this.tx_aprendizado)
            );

            // A mesma coisa só que para as bias
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

    /**
     * Ativa os valores de uma certa matriz
    */
    public double[][] ativarValores(double[][] values) {
        double[][] resultMatriz = new double[values.length][values[0].length];
        
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                resultMatriz[i][j] = this.sigmoid(values[i][j]);
            }
        }

        return resultMatriz;
    }

    /**
     * Não consegui pensar em um nome bom pra essa função, foi mal :|
     * Ela joga os valores de uma matriz na derivada da função de ativação
     */ 
    public double[][] desativarValores(double[][] values) {
        double[][] resultMatriz = new double[values.length][values[0].length];
        
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                resultMatriz[i][j] = this.dsigmoid(values[i][j]);
            }
        }

        return resultMatriz;
    }

    /**
     * Função de ativação do valor
    */
    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(z * -1));
    }

    /**
     * Derivada daa função de ativação
    */
    private double dsigmoid(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
}
