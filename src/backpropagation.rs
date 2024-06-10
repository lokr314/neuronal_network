

use std::rc::Rc;

use crate::NeuronalNetwork;
use crate::neuronal_network::Layer;
use crate::activating_functions::sigmoid_derivative;


impl NeuronalNetwork {

    pub fn train(&mut self, inputs: Rc<&[Vec<f32>]>, targets: Rc<&[Vec<f32>]>, learning_rate: f32, batch_size: usize) {
        let num_batches = inputs.len() / batch_size;

        for batch in 0..num_batches {
            let start = batch * batch_size;
            let end = start + batch_size;

            let batch_inputs = &inputs[start..end];
            let batch_targets = &targets[start..end];

            // Initialize gradients
            let mut weight_gradients: Vec<Vec<Vec<f32>>> = self.layers.iter().map(|layer| vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()]).collect();
            let mut bias_gradients: Vec<Vec<f32>> = self.layers.iter().map(|layer| vec![0.0; layer.biases.len()]).collect();

            // Process each example in the mini-batch
            for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                let (activations, zs) = self.forward_propagation(input);

                let output_activations = &activations[activations.len() - 1];
                
                // Delta for the output layer (using the softmax derivative)
                let mut delta = Vec::with_capacity(output_activations.len());
                for i in 0..output_activations.len() {
                    delta.push(output_activations[i] - target[i]);
                }

                //println!("{:?}", &delta);

                // Update gradients for the output layer
                let last_layer_index = self.layers.len() - 1;
                self.update_gradients(&mut weight_gradients[last_layer_index], &mut bias_gradients[last_layer_index], &activations[activations.len() - 2], &delta);

                // Delta for hidden layers
                let mut delta = delta;
                for l in (0..self.layers.len() - 1).rev() {
                    let mut new_delta = vec![0.0; self.layers[l].weights.len()];
                    let next_layer = &self.layers[l + 1];
                    let sp: Vec<f32> = zs[l].iter().map(|&z| sigmoid_derivative(z)).collect();

                    for i in 0..self.layers[l].weights.len() {
                        let mut error = 0.0;
                        for j in 0..next_layer.weights.len() {
                            error += next_layer.weights[j][i] * delta[j];
                        }
                        new_delta[i] = error * sp[i];
                    }
                    delta = new_delta;
                    self.update_gradients(&mut weight_gradients[l], &mut bias_gradients[l], &activations[l], &delta);
                }
            }

            //println!("{:?}", weight_gradients);

            // Update parameters for hiden layers
            for l in 0..self.layers.len() - 1 {
                for n in 0..self.layers[l].weights.len() {
                    for c in 0..self.layers[l].weights[n].len() {
                        self.layers[l].weights[n][c] -= learning_rate * weight_gradients[l][n][c] /  self.layers[l + 1].weights[0].len() as f32;
                    }
                    self.layers[l].biases[n] -= learning_rate * bias_gradients[l][n] / self.layers[l].weights[n].len() as f32;
                }
            }

            // Update parameters for output layer
            let l = self.layers.len() - 1;
            for n in 0..self.layers[l].weights.len() {
                for j in 0..self.layers[self.layers.len() - 1].weights[0].len() {
                    self.layers[l].weights[n][j] -= learning_rate * weight_gradients[l][n][j] /  self.layers[l].weights[n].len() as f32;
                }
                self.layers[l].biases[n] -= learning_rate * bias_gradients[l][n] / self.layers[l].weights[n].len() as f32;
            }
        }
    }

    pub fn update_gradients(&self, weight_gradients: &mut Vec<Vec<f32>>, bias_gradients: &mut Vec<f32>, input: &Vec<f32>, delta: &Vec<f32>) {
        for i in 0..weight_gradients.len() {
            for j in 0..weight_gradients[i].len() {
                weight_gradients[i][j] += delta[i] * input[j];
            }
            bias_gradients[i] += delta[i];
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