package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Main {
    double[] weight;

    double bias;

    double learning_rate = 1e-5;

    public static void main(String[] args) {
//        double[][] x_data = {
//                {73., 80., 75.},
//                {93., 88., 93.},
//                {89., 91., 90.},
//                {96., 98., 100.},
//                {73., 66., 70.},
//                {53., 46., 55.}
//        };
//
//        double[][] y_data = {{152.}, {185.}, {180.}, {196.}, {142.}, {101.}};

        Scanner sc = new Scanner(System.in);

        double learning_rate = 0.0;
        int end_step = 0, print_step = 0;
        String path = "", test_path = "";

        System.out.print("학습률 설정 : ");
        learning_rate = sc.nextDouble();

        System.out.print("학습 횟수 : ");
        end_step = sc.nextInt();

        System.out.print("학습 로그 출력 단위 : ");
        print_step = sc.nextInt();

        System.out.print("학습 파일 위치 (.vcs) : ");
        path = sc.next();

        System.out.print("학습 후 Test 파일 위치 (.vcs) : ");
        test_path = sc.next();

        Main main = new Main();

        Map<String, double[][]> map = main.loadCSV(path, test_path);

        double[][] x_data = map.get("x"), y_data = map.get("y"), test_data = map.get("test");

        main.initialSettings(learning_rate, x_data[0].length);
        main.learning(x_data, y_data, end_step, print_step, test_data);
    }

    private void initialSettings(double learning_rate, int weight_length) {
        Random random = new Random();
        weight = new double[weight_length];

        for (int i = 0; i < weight_length; i++) {
            weight[i] = random.nextDouble();
        }

        bias = 0.0;

        this.learning_rate = learning_rate;
    }

    private void learning(double[][] x_data, double[][] y_data, int end_step, int print_step, double[][] test_data) {
        for (int step = 1; step <= end_step; step++) {
            double[][] hypothesis = hypothesis(x_data, y_data[0].length);
            double cost = cost(hypothesis, y_data);

            gradientDescent(hypothesis, x_data, y_data);

            if (step % print_step == 0) {
                System.out.println("Step: " + step + " - Cost: " + cost);
            }
        }
        System.out.println("===============================");
        double[][] predictedOutput = hypothesis(test_data, y_data[0].length);

        for (double[] data : predictedOutput)
            System.out.println("Predicted data: " + Arrays.toString(data));
        System.out.println("weight : " + Arrays.toString(weight) + ", bias : " + bias + ", cost : " + cost(hypothesis(x_data, y_data[0].length), y_data));
    }

    private void gradientDescent(double[][] hypothesis, double[][] x_data, double[][] y_data) {
        double[] w_gradients = new double[weight.length];
        double b_gradient = 0.0;

        // 최적의 W와 b를 찾기 위해 cost를 미분한 식을 응용하여 구현
        for (int i = 0; i < x_data.length; i++) {
            double error = hypothesis[i][0] - y_data[i][0];
            for (int j = 0; j < x_data[i].length; j++) {
                w_gradients[j] += (error * x_data[i][j]) / (double) x_data.length;
            }
            b_gradient += error / (double) x_data.length;
        }

        for (int i = 0; i < weight.length; i++) {
            weight[i] -= learning_rate * w_gradients[i];
        }
        bias -= learning_rate * b_gradient;
    }

    private double cost(double[][] hypothesis, double[][] y_data) {
        double data = 0.0, var = 0.0;

        // 행열을 이용한 cost/loss 함수의 수식을 for문을 사용하여 구현
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

        // 행열을 응용한 hypothesis의 수식을 for문을 사용하여 구연
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

    private Map<String, double[][]> loadCSV(String path, String test_path) {
        Map<String, double[][]> map = null;

        try (
                BufferedReader reader = new BufferedReader(new FileReader(path));
                BufferedReader test_reader = new BufferedReader(new FileReader(test_path))
        ) {
            String line;

            List<double[]> x_list = new ArrayList<>();
            List<double[]> y_list = new ArrayList<>();
            List<double[]> test_list = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                double[] x = new double[fields.length - 1];

                for (int i = 0; i < x.length; i++) {
                    x[i] = Double.parseDouble(fields[i]);
                }

                x_list.add(x);
                y_list.add(new double[] {Double.parseDouble(fields[fields.length - 1])});
            }

            // test data load
            while ((line = test_reader.readLine()) != null) {
                String[] fields = line.split(",");
                double[] test = new double[fields.length];

                for (int i = 0; i< test.length; i++) {
                    test[i] = Double.parseDouble(fields[i]);
                }

                test_list.add(test);
            }

            map = new HashMap<>();

            map.put("x", x_list.toArray(new double[x_list.size()][x_list.get(0).length]));
            map.put("y", y_list.toArray(new double[y_list.size()][y_list.get(0).length]));
            map.put("test", test_list.toArray(new double[test_list.size()][test_list.get(0).length]));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }
}
