use std::fmt::Debug;



#[derive(Debug)]
pub struct NeuronalNetwork {
    pub layout: Vec<u16>,
    layers: Vec<Layer>
}

impl NeuronalNetwork {
    pub fn new(layout: Vec<u16>) -> Self {
        let mut layers = Vec::with_capacity(layout.len() - 1);

        for i in 1..layout.len() {
            layers.push(Layer::new(layout[i - 1], layout[i]));
        }

        NeuronalNetwork { layout, layers }
    }
}


#[derive(Debug)]
struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>
}

impl Layer {
    fn new(inputs: u16, neurons: u16) -> Self {
        let weights: Vec<Vec<f32>> = vec![vec![0.0; inputs as usize]; neurons as usize];
        let biases = vec![0.0; neurons as usize];

        Layer { weights, biases }
    }
}