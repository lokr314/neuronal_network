#[allow(unused_imports)]
mod tests {

    use crate::NeuronalNetwork;

    //NeuronalNetwork
    //Diese test function werden nicht bei einem normale compilen mit compiled sonder nur wenn man die teste ausführt
    //assert_eq! vergleicht die beiden parameter und lässt den test fehlschlagen wen sie nicht exact gleich sind
    #[test]
    fn initialisation_and_layout() {
        let neuronal_network = NeuronalNetwork::new(vec![2, 3, 3, 2]);
        assert_eq!(vec![2, 3, 3, 2], neuronal_network.get_layout());
    }
}