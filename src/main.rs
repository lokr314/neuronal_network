pub mod neuronal_network;
pub mod tests;
pub mod activating_functions;
pub mod random_changes_learning;
pub mod backpropagation;
pub mod loss_functions;

use std::{io::Write, rc::Rc};

use neuronal_network::NeuronalNetwork;


use rust_mnist::Mnist;


fn main() {
    // Lade das MNIST-Datenset
    let (trn_img, trn_lbl, test_img, test_lbl) = load_mnist();

    let layout = vec![784, 20, 10];
    let mut neural_network = NeuronalNetwork::new_random(layout.clone());

    let epochs = 1000;
    let learning_rate = 0.01;
    let batch_size = 10;

    let mut input_vec = Vec::with_capacity(test_img.len() / 784);
    let mut target_vec = Vec::with_capacity(trn_lbl.len());

    for i in 0..trn_img.len() / 784 {
        input_vec.push(trn_img[i * 784..(i + 1) * 784].to_vec());
        target_vec.push(one_hot_encoding(trn_lbl[i] as usize, 10));
    }

    let inputs: Rc<&[Vec<f32>]> = Rc::new(&input_vec);
    let targets: Rc<&[Vec<f32>]> = Rc::new(&target_vec);

    for _ in 0..epochs {
        neural_network.train(inputs.clone(), targets.clone(), learning_rate, batch_size);

        let mut correct_c: u32 = 0;
        let mut total_loss = 0.0;
        for i in 0..test_img.len() / 784 {
            let inputs = &test_img[i * 784..(i + 1) * 784];
            let targets = one_hot_encoding(test_lbl[i] as usize, 10);
            let (loss, correct) = neural_network.test(inputs.to_vec(), targets);
            total_loss = total_loss * 0.95 + 0.05 * loss;
            if correct {
                correct_c += 1;
            }
        }
        total_loss /= test_img.len() as f32;
        println!("Loss: {}, Accuracy: {:.2}%", total_loss, (correct_c as f32 / (test_img.len() / 784) as f32) * 100.);
        std::io::stdout().flush().unwrap();
    }

}

fn load_mnist() -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
    let mnist = Mnist::new("C:/Users/Uwe Kuhlmann/Desktop/Datasets/MNIST/");

    let images = mnist.train_data.into_iter().flatten().map(|x| x as f32 / 256.0).collect::<Vec<f32>>();
    let labels = mnist.train_labels;

    let test_images = mnist.test_data.into_iter().flatten().map(|x| x as f32 / 256.0).collect::<Vec<f32>>();
    let test_labels = mnist.test_labels;

    (images, labels, test_images, test_labels)
}

fn one_hot_encoding(label: usize, num_classes: usize) -> Vec<f32> {
    let mut encoding = vec![0.0; num_classes];
    encoding[label] = 1.0;
    encoding
}