import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.
       back.Backpropagation;

public class HelloWorld {

    public static double XOR_INPUT[][] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };

    public static double XOR_IDEAL[][] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    public static void main(final String[] args) {
        // Create a neural network
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,
                                        true,
                                        2));
        network.addLayer(new BasicLayer(new ActivationSigmoid(),
                                        true,
                                        2));
        network.addLayer(new BasicLayer(new ActivationSigmoid(),
                                        false,1));
        network.getStructure().finalizeStructure();

        // Set weights.
        // (fromLayer, fromNeuron, toNeuron, value)

        // I-H1
        network.setWeight(0,0,0, 0.1000); // w1
        network.setWeight(0,1,0, 0.2000); // w2
        network.setWeight(0,2,0, 0.3000); // w3 (b1)

        // I-H2
        network.setWeight(0,0,1, 0.4000); // w4
        network.setWeight(0,1,1, 0.5000); // w5
        network.setWeight(0,2,1, 0.6000); // w6 (b1)

        // H-O
        network.setWeight(1,0,0, 0.7000); // w7
        network.setWeight(1,1,0, 0.8000); // w8
        network.setWeight(1,2,0, 0.9000); // w9 (b2)

        // Create the training data
        MLDataSet trainingSet =
            new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        // Train the Neural Network
        final Backpropagation train =
            new Backpropagation(network, trainingSet, 0.7, 0.3);
	train.setThreadCount(1);
        train.fixFlatSpot(false);

        double targetMSE = 0.00000001;
        System.out.println("... Training until MSE <= " +
                String.format("%.17f", targetMSE) +
                " please wait ...");

        int epoch = 0;
        do {
            train.iteration();

            // NOTE: Encog3 default error is MSE
            if((epoch % 10000000) == 0) {
                System.out.println("MSE at epoch " +
                        String.format("%010d",epoch) + " = " +
                        String.format("%.17f",train.getError()));
            }

            ++epoch;

	} while(train.getError() > targetMSE);
        train.finishTraining();

        System.out.println("");
        System.out.println("--- TRAINING COMPLETED ---");
        System.out.println("Total epochs = "+epoch);
        System.out.println("MSE          = "+
                String.format("%.17f", train.getError()));

        // Test the neural network
        System.out.println("");
        System.out.println("--- TEST ---");
        for(MLDataPair pair: trainingSet) {
            final MLData output = network.compute(pair.getInput());
            System.out.println(
                String.format("%d", (int)pair.getInput().getData(0)) +
                " XOR " +
                String.format("%d", (int)pair.getInput().getData(1)) +
                " = " +
                String.format("%.17f", output.getData(0)) +
                " (desired output = " +
                String.format("%.17f", pair.getIdeal().getData(0))+")");
        }

        Encog.getInstance().shutdown();
    }
}

