use crate::activation::ActivationFunction;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

#[derive(Debug)]
pub struct Layer {
    weights: Array2<f64>,
    biases: Array2<f64>,
}

impl Layer {
    fn randomized(inputs: usize, elements: usize, has_bias: bool) -> Self {
        let biases = if has_bias {
            Array2::random((elements, 1), StandardNormal)
        } else {
            Array2::zeros((elements, 1))
        };

        Layer {
            weights: Array2::random((elements, inputs), StandardNormal),
            biases,
        }
    }

    fn outputs<F: ActivationFunction>(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let prod = self.weights.dot(inputs);
        let res: Array2<f64> = &prod + &self.biases;
        let res = res.mapv(F::value);
        res
    }
}

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn randomized(layer_sizes: &[usize]) -> Network {
        Network {
            layers: layer_sizes
                .iter()
                .zip(layer_sizes.iter().skip(1))
                .map(|(a, b)| Layer::randomized(*a, *b, true))
                .collect(),
        }
    }

    pub fn forward<F: ActivationFunction>(&self, inputs: &Array2<f64>) -> Array2<f64> {
        self.layers
            .iter()
            .fold(inputs.clone(), |acc, layer| layer.outputs::<F>(&acc))
    }
}

mod tests {
    use crate::activation::Sigmoid;
    use crate::network::{Layer, Network};
    use ndarray::arr2;

    #[test]
    fn layer() {
        let layer = Layer::randomized(3, 1, false);
        let inputs = arr2(&[[1.0], [2.0], [3.0]]);
        let outputs = layer.outputs::<Sigmoid>(&inputs);
        println!("{:?}", outputs);
    }

    #[test]
    fn network() {
        let network = Network::randomized(&[2, 3, 1]);
        let inputs = arr2(&[[1.0], [2.0]]);
        let outputs = network.forward::<Sigmoid>(&inputs);
        println!("{:?}", outputs);
    }
}
