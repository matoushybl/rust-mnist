use crate::activation::{ActivationFunction, Sigmoid};
use crate::dataset::load_dataset;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray::{arr1, Array1, Array2, ShapeBuilder};
use ndarray_rand::RandomExt;
use rand::distributions::{Standard, Uniform};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::Normal;
use std::convert::TryInto;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{ErrorKind, Read};

mod activation;
mod cost;
mod dataset;
mod image;

// fn forward<T: ActivationFunction>() -> Array2<f64> {}

fn main() {
    let mut training_dataset = load_dataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        .expect("Failed to load training dataset.");
    let testing_dataset = load_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        .expect("Failed to load testing dataset.");

    let layer_sizes = vec![2, 3, 1];
    let biases: Vec<Array2<f64>> = layer_sizes
        .iter()
        .skip(1)
        .map(|y| Array2::random((*y, 1), Normal::new(0.0, 1.0).unwrap()))
        .collect();

    let weights: Vec<Array2<f64>> = layer_sizes
        .windows(2)
        .filter(|elements| elements.len() > 1)
        .map(|elements| Array2::random((elements[1], elements[0]), Normal::new(0.0, 1.0).unwrap()))
        .collect();

    // forward result calculation
    let inputs = Array2::random((layer_sizes[0], 1), Normal::new(0.0, 1.0).unwrap());

    let result: Array2<f64> =
        weights
            .clone()
            .iter()
            .zip(biases.clone())
            .fold(inputs, |acc, (w, b)| {
                let prod: Array2<f64> = w.dot(&acc);
                let sum: Array2<f64> = &prod + &b;
                sum.mapv(Sigmoid::value)
            });
    // let result: Array2<f64> = weights[0].dot(&acc);
    // let result = result.mapv(sigmoid);

    println!("{}", result);

    // stochastic gradient descent
    let epochs = 100;
    let batch_size = 100usize;
    let learning_rate = 0.0001f64;

    let mut random = thread_rng();
    for epoch in 0..epochs {
        training_dataset.shuffle(&mut random);

        for batch in training_dataset.chunks(batch_size) {
            // update each batch
            // calculate gradient ( nabla )
            let nabla_biases: Vec<Array2<f64>> = biases
                .clone()
                .iter()
                .map(|bias| Array2::zeros(bias.raw_dim()))
                .collect();

            let nabla_weights: Vec<Array2<f64>> = weights
                .clone()
                .iter()
                .map(|weight| Array2::zeros(weight.raw_dim()))
                .collect();
            use std::iter::Iterator;
            for image in batch {
                // backpropagate
                let output =
                    Array2::from_shape_fn((*layer_sizes.last().unwrap(), 1), |(row, col)| {
                        if row == image.label as usize {
                            1.0
                        } else {
                            0.0
                        }
                    });

                let nabla_biases: Vec<Array2<f64>> = biases
                    .clone()
                    .iter()
                    .map(|bias| Array2::zeros(bias.raw_dim()))
                    .collect();

                let nabla_weights: Vec<Array2<f64>> = weights
                    .clone()
                    .iter()
                    .map(|weight| Array2::zeros(weight.raw_dim()))
                    .collect();
            }
            // TODO maybe validate on small subset of test data and print results to see if training is successful
        }
    }
}
