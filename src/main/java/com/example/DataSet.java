package com.example;

import java.util.List;

public class DataSet {

    public List<double[][]> images;
    public double[][] labels;

    public DataSet(List<double[][]> images, double[][] labels) {
        this.images = images;
        this.labels = labels;
    }
}
