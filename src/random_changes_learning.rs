use crate::neuronal_network::NeuronalNetwork;
use rand::Rng;
use crate::loss_functions::mean_squared_error;

impl NeuronalNetwork {
    //Sehr inefficiente training funktion die auch sehr ineffectiv ist
    pub fn train_with_random_changes(&mut self, inputs: Vec<f32>, targets: Vec<f32>, learning_rate: f32, iterations: usize) -> f32 {
        let prediction = self.feed_forward(inputs.clone());
        let mut current_loss = mean_squared_error(prediction, targets.clone());
        
        let mut rng = rand::thread_rng();

        for _ in 0..iterations {
            // Speichere aktuelle Gewichte und Biases
            let random_layer: usize = rng.gen_range(0..self.layers.len());
            let random_neuron: usize = rng.gen_range(0..self.layers[random_layer].weights.len());
            
            let saved_neuron = self.layers[random_layer].weights[random_neuron].clone();
            let saved_bias = self.layers[random_layer].biases[random_neuron].clone();

            // Ändere die Gewichte und Biases zufällig
            for weight in &mut self.layers[random_layer].weights[random_neuron] {
                *weight += rng.gen_range(-learning_rate..learning_rate);
            }
            self.layers[random_layer].biases[random_neuron] += rng.gen_range(-learning_rate..learning_rate) * 0.01;

            // Berechne den neuen Verlust#
            let new_predictions = self.feed_forward(inputs.clone());
            let new_loss = mean_squared_error(new_predictions, targets.clone());

            // Wenn der neue Verlust besser ist, akzeptiere die Änderungen
            // Wenn nicht, setze die ursprünglichen Gewichte und Biases zurück
            if new_loss < current_loss {
                current_loss = new_loss;
            } else {
                self.layers[random_layer].weights[random_neuron] = saved_neuron;
                self.layers[random_layer].biases[random_neuron] = saved_bias;
            }
            //println!("current_loss: {}", current_loss);
        }

        current_loss
    }
}