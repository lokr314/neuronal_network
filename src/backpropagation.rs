

use crate::NeuronalNetwork;
use crate::neuronal_network::Layer;
use crate::activating_functions::{leaky_relu, leaky_relu_derivative};


impl NeuronalNetwork {

    pub fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>, learning_rate: f32) {
        let mut activations = vec![inputs.clone()];
        let mut zs = Vec::with_capacity(self.layers.len());

        let mut activation = inputs;
        for layer in 0..self.layers.len() - 1 {
            let z = self.layers[layer].feed_forward(&activation);
            zs.push(z.clone());
            activation = z;
            activations.push(activation.clone());
        }

        activations.push(self.layers[self.layers.len() - 1].feed_output(&activation));

        let mut delta = Vec::with_capacity(activations[activations.len() - 1].len());
        {
            let output_activations = activations.last().unwrap();
            for i in 0..output_activations.len() {
                delta.push((output_activations[i] - targets[i]).powi(2));
            }
        }

        let last_activation = activations[activations.len() - 2].clone();
        self.layers.last_mut().unwrap().update_parameters(&last_activation, &delta, learning_rate);

        for l in (0..self.layers.len() - 1).rev() {
            let mut new_delta = vec![0.0; self.layers[l].weights.len()];
            let next_layer = &self.layers[l + 1];
            let sp: Vec<f32> = zs[l].iter().map(|&z| leaky_relu_derivative(z)).collect();

            for i in 0..self.layers[l].weights.len() {
                let mut error = 0.0;
                for j in 0..next_layer.weights.len() {
                    error += next_layer.weights[j][i] * delta[j];
                }
                new_delta[i] = error * sp[i];
            }
            delta = new_delta;
            self.layers[l].update_parameters(&activations[l], &delta, learning_rate);
        }
    }

}

impl Layer {
    pub fn update_parameters(&mut self, input: &[f32], delta: &[f32], learning_rate: f32) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= learning_rate * delta[i] * input[j];
            }
            self.biases[i] -= learning_rate * delta[i];
        }
    }
}