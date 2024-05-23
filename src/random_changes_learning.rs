use crate::neuronal_network::NeuronalNetwork;
use rand::Rng;

impl NeuronalNetwork {
    //Sehr inefficiente training funktion die auch sehr ineffectiv ist
    pub fn train_with_random_changes(&mut self, inputs: Vec<f32>, targets: Vec<f32>, learning_rate: f32, iterations: usize) -> f32 {
        let current_loss = self.calculate_loss(inputs.clone(), targets.clone());
        let mut new_loss: f32 = 0.0;
        
        let mut rng = rand::thread_rng();

        // Speichere aktuelle Gewichte und Biases
        let random_layer: usize = rng.gen_range(0..self.layers.len() as _);
        let random_neuron:usize = rng.gen_range(0..self.layers[random_layer].weights.len() as _);

        let saved_neuron = self.layers[random_layer].weights[random_neuron].clone();
        let saved_bias = self.layers[random_layer].biases[random_neuron];

        for _ in 0..iterations {
            for weight in &mut self.layers[random_layer].weights[random_neuron] {
                *weight += rng.gen_range(-learning_rate..learning_rate);
            }
    
            self.layers[random_layer].biases[random_neuron] += rng.gen_range(-learning_rate..learning_rate);
    
            let loss = self.calculate_loss(inputs.clone(), targets.clone());
    
            // Vergleiche die Verluste
            if current_loss >= loss {
                for weight in 0..self.layers[random_layer].weights[random_neuron].len() {
                    self.layers[random_layer].weights[random_neuron][weight] = saved_neuron[weight];
                }
        
                self.layers[random_layer].biases[random_neuron] = saved_bias;
            } else {
                new_loss = loss;
            }
        }

        new_loss
    }
}