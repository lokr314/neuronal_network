
pub fn leaky_relu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 0.01 }
}
pub fn leaky_relu_derivative(x: f32) -> f32 {
    if x >= 0.0 { x } else { x * 100.0 }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}