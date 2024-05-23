use std::{alloc::Layout, fmt::Debug, mem::size_of_val};



#[derive(Debug)]
pub struct NeuronalNetwork {
    layers: Vec<Layer>
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
}


#[derive(Debug)]
struct Layer {
    //[neuronen[connections]]
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