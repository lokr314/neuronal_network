mod neuronal_network;
mod tests;
mod activating_functions;
mod random_changes_learning;

use neuronal_network::NeuronalNetwork;

use rand::Rng;


fn main() {
    let start_time = std::time::Instant::now();
    let layout = vec![1, 20, 1];
    let mut nn = NeuronalNetwork::new(layout);
    let mut rng = rand::thread_rng();
    let mut loss = 0.0;

    for _ in 0..1000000000 {
        let random_number: f32 = rng.gen_range(-1.0..1.0);
        let sin = random_number.sin();
        loss = 0.999 * loss + 0.001 * nn.train_with_random_changes(vec![random_number], vec![sin], 0.0001, 20);
        print!("\r{}", loss);
    }

    println!("{:?} : {:?}", nn.get_layout(), start_time.elapsed());
}
