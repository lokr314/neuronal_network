#[allow(dead_code)]
pub fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { x * 0.01 }
}
pub fn leaky_relu_derivative(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01 }
}
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}
pub fn relu_derivative(z: f32) -> f32 {
    if z > 0.0 { 1.0 } else { 0.0 }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

pub fn softmax(input: &[f32]) -> Vec<f32> {
    // Berechne den maximalen Wert im Vektor (zur numerischen Stabilisierung)
    let mut max_value = 0.0;

    for i in 0..input.len() {
        if !input[i].is_nan() {
            if input[i] > max_value {
                max_value = input[i];
            }
        }
    }
    
    // Berechne die exponentiellen Werte und deren Summe
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_value).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    
    // Teile jeden exponentiellen Wert durch die Summe der exponentiellen Werte
    exps.iter().map(|&exp| exp / sum_exps).collect()
}

pub fn softbest(input: &[f32]) -> Vec<f32> {
    // Berechne den maximalen Wert im Vektor (zur numerischen Stabilisierung)
    let mut max_value = 0.0;

    for i in 0..input.len() {
        if !input[i].is_nan() {
            if input[i].abs() > max_value {
                max_value = input[i];
            }
        }
    }
    
    // Berechne die exponentiellen Werte und deren Summe
    let exps: Vec<f32> = input.iter().map(|&x| expi(x - max_value)).collect();
    let sum_exps: f32 = exps.iter().sum();
    
    // Teile jeden exponentiellen Wert durch die Summe der exponentiellen Werte
    let softmax: Vec<f32> = exps.iter().map(|&exp| exp / sum_exps).collect();
    let mean = softmax.iter().copied().sum::<f32>() / softmax.len() as f32;
    softmax.iter().map(|&value| value - mean).collect()
}

fn expi(x: f32) -> f32 {
    std::f32::consts::E.powf(x) * x.signum()
}