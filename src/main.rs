use std::vec;

use neuronal_network::NeuronalNetwork;
mod neuronal_network;

fn main() {
    let v = vec![ 0, 3, 4];
    let k = NeuronalNetwork::new(v);
}
