use rand::Rng;

use crate::activating_functions::leaky_relu;

#[derive(Debug)]
pub struct NeuronalNetwork {
    pub layers: Vec<Layer>
}

impl NeuronalNetwork {
    pub fn new(layout: Vec<u16>) -> Self {
        let mut layers = Vec::with_capacity(layout.len() - 1);
        for i in 1..layout.len() {
            layers.push(Layer::new(layout[i - 1], layout[i]));
        }
        NeuronalNetwork { layers }
    }

    pub fn new_random(layout: Vec<u16>) -> Self {
        let mut layers = Vec::with_capacity(layout.len() - 1);
        for i in 1..layout.len() {
            layers.push(Layer::new_random(layout[i - 1], layout[i]));
        }
        NeuronalNetwork { layers }
    }

    pub fn get_layout(&self) -> Vec<usize> {

        //Der input layer ist nicht in layers vorhanden
        let mut layout = Vec::with_capacity(self.layers.len() + 1);
        layout.push(self.layers[0].weights[0].len());

        //Für jede weitere schicht wird die anzahl der neuronen genommen
        for layer in &self.layers {
            layout.push(layer.weights.len());
        }
        layout
    }

    //Output funktion fehlt
    pub fn feed_forward(&self, input: Vec<f32>) -> Vec<f32> {
        let mut input = input;
        for layer in 0..self.layers.len() - 1 {
            input = self.layers[layer].feed_forward(&input);
        }
        self.layers.last().unwrap().feed_output(&input)
    }

    pub fn test(&self, inputs: Vec<f32>, targets: Vec<f32>) -> (f32, bool) {
        let output = self.feed_forward(inputs);
        let mut error = 0.0;
        for i in 0..output.len() {
            error += (output[i] - targets[i]).powi(2);
        }
        let mut greatest_i = 0.0;
        let mut index = 0;
        for i in 0..output.len() {
            if output[i] > greatest_i {
                greatest_i = output[i];
                index = i;
            }
        }
        (error / output.len() as f32, targets[index] == 1.0)
    }

}


#[derive(Debug, Clone)]
pub struct Layer {
    //[neuronen[connections]]
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>
}

impl Layer {
    pub fn new(inputs: u16, neurons: u16) -> Self {
        let weights: Vec<Vec<f32>> = vec![vec![0.0; inputs as usize]; neurons as usize];
        let biases = vec![0.0; neurons as usize];

        Layer { weights, biases }
    }

    pub fn new_random(inputs: u16, neurons: u16) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<Vec<f32>> = (0..neurons)
            .map(|_| (0..inputs).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let biases: Vec<f32> = (0..neurons).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Layer { weights, biases }
    }

    pub fn feed_forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.weights.len());

        for neuron in 0..self.weights.len() {
            let mut neuron_output = self.biases[neuron];

            for weight in 0..self.weights[0].len() {
                neuron_output += input[weight] * self.weights[neuron][weight];
            }

            output.push(leaky_relu(neuron_output));
        }

        output
    }

    pub fn feed_output(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.weights.len());

        for neuron in 0..self.weights.len() {
            let mut neuron_output = self.biases[neuron];

            for weight in 0..self.weights[0].len() {
                neuron_output += input[weight] * self.weights[neuron][weight];
            }

            output.push(neuron_output.min(1.0));
        }

        output
    }
}