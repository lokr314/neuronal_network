use std::vec;

use neuronal_network::NeuronalNetwork;
mod neuronal_network;

fn main() {
    let v = vec![2, 3, 3, 5, 2];
    let k = NeuronalNetwork::new(v);

    println!("{:?}", k.get_layout());
}
