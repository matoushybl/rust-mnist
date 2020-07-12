pub trait ActivationFunction {
    fn value(z: f64) -> f64;
    fn derivative(z: f64) -> f64;
}

pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn value(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn derivative(z: f64) -> f64 {
        unimplemented!()
    }
}
