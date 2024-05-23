use crate::activating_functions::sigmoid;

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
    pub fn feed_forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            input = layer.feed_forward(input);
        }
        input
    }

    pub fn calculate_loss(&self, inputs: Vec<f32>, targets: Vec<f32>) -> f32 {
        let predictions = self.feed_forward(inputs);
        let mut sum = 0.0;
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            sum += (pred - target).powi(2);
        }
        sum / predictions.len() as f32
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

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.weights.len());

        for neuron in 0..self.weights.len() {
            let mut neuron_output = self.biases[neuron];

            for weight in 0..self.weights[0].len() {
                neuron_output += inputs[weight] * self.weights[neuron][weight];
            }

            output.push(sigmoid(neuron_output));
        }

        output
    }
}