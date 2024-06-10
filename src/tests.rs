#[allow(unused_imports)]
mod tests {

    use crate::{NeuronalNetwork, activating_functions};

    //NeuronalNetwork
    //Diese test function werden nicht bei einem normale compilen mit compiled sonder nur wenn man die teste ausführt
    //assert_eq! vergleicht die beiden parameter und lässt den test fehlschlagen wen sie nicht exact gleich sind
    #[test]
    fn initialisation_and_layout() {
        let neuronal_network = NeuronalNetwork::new(vec![2, 3, 3, 2]);
        assert_eq!(vec![2, 3, 3, 2], neuronal_network.get_layout());
    }

    #[test]
    fn random_init() {
        let neuronal_network = NeuronalNetwork::new_random(vec![2, 3, 3, 2]);
        println!("{:?}", neuronal_network);
    }

    #[test]
    fn softmax() {
        let input = vec![1.0, 1.0, 2.0, 1.0, 1.0];
    
        let result = activating_functions::softmax(&input);
        println!("{:?}", result);
        println!("{:?}", activating_functions::softmax_derivative(&result))
    }

    #[test]
    fn backpropagation() {
        let layout = vec![2, 3, 2];
        
    }
}