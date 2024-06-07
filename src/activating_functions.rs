
pub fn leaky_relu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 0.01 }
}
pub fn leaky_relu_derivative(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 100.0 }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softmax(input: &[f32]) -> Vec<f32> {
    // Berechne den maximalen Wert im Vektor (zur numerischen Stabilisierung)
    let max_value = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    // Berechne die exponentiellen Werte und deren Summe
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_value).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    
    // Teile jeden exponentiellen Wert durch die Summe der exponentiellen Werte
    exps.iter().map(|&exp| exp / sum_exps).collect()
}

pub fn softmax_derivative(input: &[f32]) -> Vec<Vec<f32>> {
    let s = softmax(input);
    let n = s.len();
    let mut jacobian = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                jacobian[i][j] = s[i] * (1.0 - s[j]);
            } else {
                jacobian[i][j] = -s[i] * s[j];
            }
        }
    }

    jacobian
}