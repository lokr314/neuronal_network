mod neuronal_network;
mod tests;
mod activating_functions;
mod random_changes_learning;

use neuronal_network::NeuronalNetwork;

use rand::Rng;


fn main() {
    let layout = vec![1, 10, 1];
    let mut nn = NeuronalNetwork::new_random(layout);
    let mut rng = rand::thread_rng();
    let mut loss = nn.train_with_random_changes(vec![0.0], vec![0.0], 1.0, 20);

    for _ in 0..1_000_000_0 {
        let random_number: f32 = rng.gen_range(-10..10) as f32 / 10.0;
        let target = {
            if random_number < 0.3 && random_number > -0.3 {
                0.0
            } else {
                1.0
            }
        };
        loss = 0.95 * loss + 0.05 * nn.train_with_random_changes(vec![random_number], vec![target], 0.1, 20);
        print!("\r{}", loss);
    }


    print!("\n");
    for _ in 0..10 {
        let random_number: f32 = rng.gen_range(-10..10) as f32 / 10.0;
        let target = {
            if random_number < 0.3 && random_number > -0.3 {
                0.0
            } else {
                1.0
            }
        };
        println!("value: {}, target: {}, NN: {}", random_number, target, nn.feed_forward(vec![random_number])[0]);
    }
}
