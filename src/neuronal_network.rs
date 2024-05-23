
#[derive(Debug)]
pub struct NeuronalNetwork {
    pub layout: Vec<u16>
}

impl NeuronalNetwork {
    pub fn new(layout: Vec<u16>) -> Self {
        NeuronalNetwork { layout }
    }
}