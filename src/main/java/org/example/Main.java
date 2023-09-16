package org.example;

import java.util.Arrays;

public class Main {
    double[] weight = {0., 0., 0.};

    double bias = 1.0;

    double learning_rate = 1e-5;

    public static void main(String[] args) {
        double[][] x_data = {
                {73., 80., 75.},
                {93., 88., 93.},
                {89., 91., 90.},
                {96., 98., 100.},
                {73., 66., 70.},
                {53., 46., 55.}
        };

        double[][] y_data = {{152.}, {185.}, {180.}, {196.}, {142.}, {101.}};

        Main main = new Main();

        for (int step = 1; step <= 10000; step++) {
            double[][] hypothesis = main.hypothesis(x_data, y_data[0].length);
            double cost = main.cost(hypothesis, y_data);

            main.gradientDescent(hypothesis, x_data, y_data);

            if (step % 1000 == 0) {
                System.out.println("Step: " + step + " - Cost: " + cost);
            }
        }

        double[][] newInput = {{100., 70., 101.}};
        double[][] predictedOutput = main.hypothesis(newInput, y_data[0].length);

        for (double[] data : predictedOutput)
            System.out.println("Predicted data: " + Arrays.toString(data));
    }

    private void gradientDescent(double[][] hypothesis, double[][] x_data, double[][] y_data) {
        double[] w_gradients = new double[weight.length];
        double b_gradient = 0.0;


        for (int i = 0; i < x_data.length; i++) {
            for (int j = 0; j < x_data[i].length; j++) {
                double error = hypothesis[i][0] - y_data[i][0];
                w_gradients[j] += (2.0 / x_data.length) * error * x_data[i][j];
            }
            b_gradient += (2.0 / x_data.length) * (hypothesis[i][0] - y_data[i][0]);
        }

        for (int i = 0; i < weight.length; i++) {
            weight[i] -= learning_rate * w_gradients[i];
        }
        bias -= learning_rate * b_gradient;
    }

    private double cost(double[][] hypothesis, double[][] y_data) {
        double data = 0.0, var = 0.0;

        for (int i = 0; i < hypothesis.length; i++) {
            for (int j = 0; j < hypothesis[i].length; j++) {
                var = hypothesis[i][j] - y_data[i][j];
                data += (var * var) / hypothesis[i].length;
            }
        }

        return data / hypothesis.length;
    }

    private double[][] hypothesis(double[][] x_data, int y_length) {
        double[][] data = new double[x_data.length][y_length];

        for (int k = 0; k < y_length; k++) {
            for (int i = 0; i < x_data.length; i++) {
                for (int j = 0; j < x_data[i].length; j++) {
                    data[i][k] += x_data[i][j] * weight[j];
                }
                data[i][k] += bias;
            }
        }

        return data;
    }
}