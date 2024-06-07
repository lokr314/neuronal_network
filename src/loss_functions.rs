

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

pub fn error(prediction: Vec<f32>, target: Vec<f32>) -> Vec<f32> {
    if cfg!(debug_assertions) {
        if prediction.len() != target.len() {
            panic!("The vec size {} of predictions doesnt match the target vec size {}!", prediction.len(), target.len());
        }
    }
    let mut error = vec![0.0; prediction.len()];
    
    for i in 0..prediction.len() {
        error[i] = (prediction[i] - target[i]).powi(2);
    }
    error
}