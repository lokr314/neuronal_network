

pub fn mean_squared_error(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
    if cfg!(debug_assertions) {
        if predictions.len() != targets.len() {
            panic!("The vec size {} of predictions doesnt match the target vec size {}!", predictions.len(), targets.len());
        }
    }
    let mut sum = 0.0;
    for i in 0..predictions.len() {
        sum += (predictions[i] - targets[i]).powi(2);
    }
    sum / predictions.len() as f32
}