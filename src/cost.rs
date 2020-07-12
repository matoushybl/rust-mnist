pub trait CostFunction {
    fn value(prediction: f64, actual: f64) -> f64;
    fn derivative(prediction: f64, actual: f64) -> f64;
}

pub struct SimpleCost;
impl CostFunction for SimpleCost {
    fn value(prediction: f64, actual: f64) -> f64 {
        return (prediction - actual).powi(2);
    }

    fn derivative(prediction: f64, actual: f64) -> f64 {
        return 2.0f64 * (prediction - actual);
    }
}
