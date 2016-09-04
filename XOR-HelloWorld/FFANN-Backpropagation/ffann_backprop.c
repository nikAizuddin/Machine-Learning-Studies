/******************************************************************//**
 *
 *                   Feed Forward Neural Networks
 *    XOR Hello World using Backpropagation with Sigmoid Activation
 *
 * --- Network Configuration ---
 * Consists of 3 layers;
 * Input Layer: 2 input neurons, 1 bias neuron.
 * Hidden Layer: 2 hidden neurons, 1 bias neuron, Sigmoid Activation.
 * Output Layer: 1 output neurons, Sigmoid Activation.
 *
 * --- XOR Dataset ---
 * +--------+--------+---------------------------+
 * | Input1 | Input2 | Expected (Desired) Output |
 * +--------+--------+---------------------------+
 * |   0    |   0    |             0             |
 * |   1    |   0    |             1             |
 * |   0    |   1    |             1             |
 * |   1    |   1    |             0             |
 * +--------+--------+---------------------------+
 *
 * --- Features of this program ---
 * Uses "momentum" during updateWeight(), to prevent "oscillation"
 * on learning mechanism due to high learning rate (gain).
 * This purpose is to speed up the training process in some cases.
 *
 * --- Limitations of this program ---
 * 1) Unlike Encog v3.3 program, this program ignore the "Flat Spot"
 *    issues. If you want to compare resuts from this program with
 *    results from Encog v3.3, make sure you disable the "Flat Spot"
 *    feature in Encog v3.3.
 * 2) The number of neurons in Input layer, Hidden layers, and Output
 *    layer cannot be changed. This source code is not designed to do
 *    such thing.
 * 3) The initial weight values are not randomized. You have to
 *    manually set the initial weight values.
 *
 * --- Notes ---
 * I write this program because I want to make sure my calculations
 * are correct after the first two epochs.
 *
 * --- How to compile and run? ---
 * $ cc ffann_backprop.c -lm -o ffann_backprop
 * $ ./ffann_backprop
 *
 * --------------------------------------------------------------------
 *       Author: Nik Mohamad Aizuddin bin Nik Azmi
 *        Email: nik-mohamad-aizuddin@yandex.com
 * Program Name: ffann_backprop.c
 *      Version: 1.0.0
 *   C Standard: ANSI-C / C89
 *
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Uncomment this to dump network values to stdout */
/*#define DUMP_NETWORK_VALUES*/

#define NUM_OF_PATTERNS 4
#define NUM_OF_WEIGHTS 9
#define NUM_OF_INPUT_WEIGHTS 3
#define NUM_OF_OUTPUTNEURONS 1
#define NUM_OF_LAYERS 3

typedef struct weight_t {
    double value;
    double delta[NUM_OF_PATTERNS];
    double prevDelta[NUM_OF_PATTERNS]; /* required for momentum */
} weight_t;

typedef struct neuron_t {
    weight_t *inW[3];
    weight_t *outW[1];
    double    sum;
    double    output[NUM_OF_PATTERNS];
    double    delta[NUM_OF_PATTERNS];
} neuron_t;

typedef struct layer_t {
    int       totalNeurons;
    neuron_t *neuron[3]; /* Fixed to 3 neurons per layer */
} layer_t;

typedef struct dataset_t {
    double input[2];
    double desiredOutput[1];
} dataset_t;

typedef struct trainingDataset_t {
    dataset_t pattern[NUM_OF_PATTERNS];
} trainingDataset_t;

double            errorSSE    = INFINITY;
double            errorLSE    = INFINITY;
double            errorMSE    = INFINITY;
double            errorRMSE   = INFINITY;
unsigned int      totalEpochs = 0;
trainingDataset_t trainingDataset;
weight_t          weight[NUM_OF_WEIGHTS];
neuron_t          inputNeuron[2];
neuron_t          hiddenNeuron[2];
neuron_t          outputNeuron[1];
neuron_t          biasNeuron[2];
layer_t           layer[3]; /* 3 layers: input, hidden, and output */

void   init( void );
void   backpropagationTraining( double gain,
                                double momentum,
                                double targetMSE );
void   forwardPass( int p );
void   backwardPass( int p );
void   checkError( void );
void   updateWeight( double gain, double momentum );
void   showResults( void );
void   dumpNetworkValues( int epoch );
double test( double i1, double i2 );
double sigmoidActivation( double x );
double sigmoidDerivative( double x );


int main(void)
{
    /* Initialize the weight and training dataset */
    init();

    /* Train the feed forward neural network using backpropagation
     * with sigmoid activation
     */
    backpropagationTraining(0.7,          /* Learning Rate */
                            0.3,          /* Momentum */
                            0.00000001); /* Target MSE */

    /* Show results and test the trained neural network */
    showResults();

    return 0;
}


/******************************************************************//**
 * Initialize the Neural Network.
 *********************************************************************/
void init(void)
{
    int i = 0;
    int j = 0;
    int p = 0;
    int w = 0;

    /* Set weight value */
    weight[0].value = 0.1000;
    weight[1].value = 0.2000;
    weight[2].value = 0.3000;
    weight[3].value = 0.4000;
    weight[4].value = 0.5000;
    weight[5].value = 0.6000;
    weight[6].value = 0.7000;
    weight[7].value = 0.8000;
    weight[8].value = 0.9000;

    for (w = 0; w < NUM_OF_WEIGHTS; ++w) {
        for (p = 0; p < NUM_OF_PATTERNS; ++p) {
            weight[w].prevDelta[p] = 0.0000;
        }
    }

    /* Initialize the training dataset */
    trainingDataset.pattern[0].input[0] = 0.0000;
    trainingDataset.pattern[0].input[1] = 0.0000;
    trainingDataset.pattern[0].desiredOutput[0] = 0.0000;

    trainingDataset.pattern[1].input[0] = 1.0000;
    trainingDataset.pattern[1].input[1] = 0.0000;
    trainingDataset.pattern[1].desiredOutput[0] = 1.0000;

    trainingDataset.pattern[2].input[0] = 0.0000;
    trainingDataset.pattern[2].input[1] = 1.0000;
    trainingDataset.pattern[2].desiredOutput[0] = 1.0000;

    trainingDataset.pattern[3].input[0] = 1.0000;
    trainingDataset.pattern[3].input[1] = 1.0000;
    trainingDataset.pattern[3].desiredOutput[0] = 0.0000;

    /* Set biasNeuron output */
    for(i = 0; i < 2; ++i) {
        for(p = 0; p < 4; ++p) {
            biasNeuron[i].output[p] = 1.0000;
        }
    }

    /* Set inputNeuron output */
    for(i = 0; i < 2; ++i) {
        for(p = 0; p < 4; ++p) {
            inputNeuron[i].output[p] =
                    trainingDataset.pattern[p].input[i];
        }
    }

    /* Set weights for hidden neurons */
    w = 0;
    for (i = 0; i < 2; ++i) {
        for (j = 0; j < 3; ++j) {
            hiddenNeuron[i].inW[j] = &weight[w];
            ++w;
        }
    }
    hiddenNeuron[0].outW[0] = &weight[6];
    hiddenNeuron[1].outW[0] = &weight[7];

    /* Set weights for output neurons */
    for (j = 0; j < 3; ++j) {
        outputNeuron[0].inW[j] = &weight[w];
        ++w;
    }

    /* Set input layer */
    layer[0].neuron[0] = &inputNeuron[0];
    layer[0].neuron[1] = &inputNeuron[1];
    layer[0].neuron[2] = &biasNeuron[0];
    layer[0].totalNeurons = 2; /* Bias is not included */

    /* Set hidden layer */
    layer[1].neuron[0] = &hiddenNeuron[0];
    layer[1].neuron[1] = &hiddenNeuron[1];
    layer[1].neuron[2] = &biasNeuron[1];
    layer[1].totalNeurons = 2;

    /* Set output layer */
    layer[2].neuron[0] = &outputNeuron[0];
    layer[2].totalNeurons = 1;

} /* end function init() */


/******************************************************************//**
 * Perform Backpropagation Training.
 *********************************************************************/
void backpropagationTraining( double gain,
                              double momentum,
                              double targetMSE )
{
    unsigned int epoch     = 0;
    int          p         = 0;

    printf("... Training until MSE <= %.17lf, please wait ...\n",
            targetMSE);

    while (errorMSE > targetMSE) {

        /* Perform forward pass and backward pass for every pattern.
         * Results for every pattern is saved for updateWeight() use.
         */
        for (p = 0; p < NUM_OF_PATTERNS; ++p) {
            forwardPass(p);
            backwardPass(p);
        }

        /* This function does not affect the calculations of the
         * neural network. It just calculate errors, to show how
         * good the performance of the neural network has progressed
         * so far.
         */
        checkError();

        /* Display error for every 10 million epochs.
         * Training can take a long time. It is a good idea to
         * report some progress.
         */
        if(epoch % 10000000 == 0) {
            printf("MSE at epoch %010d = %.17lf\n", epoch, errorMSE); 
        }

        /* Make some adjustment to the weight values,
         * based on delta and gradients calculated from forwardPass()
         * and backwardPass().
         */
        updateWeight(gain, momentum);

#ifdef DUMP_NETWORK_VALUES
        /* Show the weight values for the first two epochs only.
         * This for study or debug purpose, to make sure the
         * calculations are correct.
         */
        dumpNetworkValues(epoch);
#endif

        ++epoch;
    } /* end while */

    /* At this point, we have done the backpropagation training */
    printf("MSE at epoch %010d = %.17lf, target MSE reached!\n",
            epoch, errorMSE); 
    totalEpochs = epoch;

} /* end function backpropagationTraining() */


/******************************************************************//**
 * Perform forward pass.
 *********************************************************************/
void forwardPass(int p)
{
    int i = 1; /* layer 2, the hidden layer */
    int n = 0;
    int w = 0;

    while (i < NUM_OF_LAYERS) {

        for(n = 0; n < layer[i].totalNeurons; ++n) {
            (layer[i].neuron[n])->sum = 0;

            /* Summation */
            for(w = 0; w < NUM_OF_INPUT_WEIGHTS; ++w) {
                (layer[i].neuron[n])->sum +=
                        (layer[i].neuron[n])->inW[w]->value *
                        (layer[i-1].neuron[w])->output[p];
            }

            /* Activation */
            (layer[i].neuron[n])->output[p] =
                    sigmoidActivation((layer[i].neuron[n])->sum);
        }

        ++i;
    }
}


/******************************************************************//**
 * Perform backward pass.
 *********************************************************************/
void backwardPass(int p)
{
    int n = 0;

    /* Find delta for output neuron */

    double diffOutput =
            - ( trainingDataset.pattern[p].desiredOutput[0] - 
            (layer[2].neuron[0])->output[p] );

    double derivative =
            sigmoidDerivative((layer[2].neuron[0])->output[p]);

    (layer[2].neuron[0])->delta[p] = diffOutput * derivative;

    /* Find delta for hidden neurons */
    for(n = 0; n < layer[1].totalNeurons; ++n) {
        (layer[1].neuron[n])->delta[p] =
                ( (layer[2].neuron[0])->delta[p] *
                  (layer[1].neuron[n])->outW[0]->value ) *
                sigmoidDerivative(
                        (layer[1].neuron[n])->output[p]);
    }
}


/******************************************************************//**
 * Sigmoid Activation function.
 *********************************************************************/
double sigmoidActivation(double x)
{
    double y = 1 / (1+exp(-x)); 
    return y;
}


/******************************************************************//**
 * Sigmoid Derivative function.
 *********************************************************************/
double sigmoidDerivative(double x)
{
    double y = x * (1 - x);
    return y;
}


/******************************************************************//**
 * Calculate SSE, LSE, MSE, and RMSE errors. But, we only use MSE.
 * SSE, LSE, and RMSE are calculated for research purpose.
 *********************************************************************/
void checkError(void)
{
    double error;
    int    n;
    int    p;

    /* SSE */
    error = 0;
    for (p = 0; p < NUM_OF_PATTERNS; ++p) {
        for (n = 0; n < NUM_OF_OUTPUTNEURONS; ++n) {
            error += pow( trainingDataset.pattern[p].desiredOutput[n] - 
                    outputNeuron[0].output[p], 2 );
        }
    }
    errorSSE = error;

    /* LSE */
    errorLSE = errorSSE / 2;

    /* MSE */
    errorMSE = errorSSE / (NUM_OF_PATTERNS * NUM_OF_OUTPUTNEURONS);

    /* RMSE */
    errorRMSE = sqrt(errorMSE);
}


/******************************************************************//**
 * Update all weight values.
 *********************************************************************/
void updateWeight(double gain, double momentum)
{
    int p = 0;
    int w = 0;
    int i = 1;
    int n = 0;

    for (i = 1; i < NUM_OF_LAYERS; ++ i) {
    for (n = 0; n < layer[i].totalNeurons; ++n) {
    for (w = 0; w < NUM_OF_INPUT_WEIGHTS; ++w) {
    for (p = 0; p < NUM_OF_PATTERNS; ++p) {

        (layer[i].neuron[n])->inW[w]->delta[p] = -
                ( gain * ( (layer[i].neuron[n])->delta[p] *
                  (layer[i-1].neuron[w])->output[p] ) )
                +
                ( momentum *
                  (layer[i].neuron[n])->inW[w]->prevDelta[p] );

        (layer[i].neuron[n])->inW[w]->value +=
                (layer[i].neuron[n])->inW[w]->delta[p];

        (layer[i].neuron[n])->inW[w]->prevDelta[p] =
                (layer[i].neuron[n])->inW[w]->delta[p];


    } /* end loop patterns */
    } /* end loop weights */
    } /* end loop neurons */
    } /* end loop layers */

} /* end function updateWeight() */


/******************************************************************//**
 * Show results and call test() to test the neural network.
 *********************************************************************/
void showResults(void)
{
    printf("\n--- TRAINING COMPLETED ---\n");
    printf("Total epochs                  = %d\n", totalEpochs);
    printf("Sum of Sqaure Errors (SSE)    = %.17lf\n", errorSSE);
    printf("Least Square Error (LSE)      = %.17lf\n", errorLSE);
    printf("Mean Square Error (MSE)       = %.17lf\n", errorMSE);
    printf("Root Mean Square Error (RMSE) = %.17lf\n", errorRMSE);

    printf("\n--- TEST ---\n");
    printf("0 XOR 0 = %.17lf (desired output = %.17lf)\n",
            test(0,0), trainingDataset.pattern[0].desiredOutput[0]);
    printf("1 XOR 0 = %.17lf (desired output = %.17lf)\n",
            test(1,0), trainingDataset.pattern[1].desiredOutput[0]);
    printf("0 XOR 1 = %.17lf (desired output = %.17lf)\n",
            test(0,1), trainingDataset.pattern[2].desiredOutput[0]);
    printf("1 XOR 1 = %.17lf (desired output = %.17lf)\n",
            test(1,1), trainingDataset.pattern[3].desiredOutput[0]);
}


/******************************************************************//**
 * Test the neural network.
 *********************************************************************/
double test(double i1, double i2)
{
    int n = 0;
    int w = 0;
    double input[3];

    input[0] = i1;
    input[1] = i2;
    input[2] = 1.0000;

    /* Hidden Layer */
    for(n = 0; n < layer[1].totalNeurons; ++n) {

        /* Summation */
        (layer[1].neuron[n])->sum = 0;
        for(w = 0; w < NUM_OF_INPUT_WEIGHTS; ++w) {
            (layer[1].neuron[n])->sum +=
                    (layer[1].neuron[n])->inW[w]->value * input[w];
        }

        /* Activation */
        (layer[1].neuron[n])->output[0] =
                sigmoidActivation((layer[1].neuron[n])->sum);
    }

    /* Output Layer */
    (layer[2].neuron[0])->sum = 0;
    for(w = 0; w < NUM_OF_INPUT_WEIGHTS; ++w) {
        (layer[2].neuron[0])->sum +=
                (layer[2].neuron[0])->inW[w]->value *
                (layer[1].neuron[w])->output[0];
    }

    /* Output Layer Sigmoid Activation */
    (layer[2].neuron[0])->output[0] =
            sigmoidActivation((layer[2].neuron[0])->sum);

    return (layer[2].neuron[0])->output[0];
}


/******************************************************************//**
 * Dump network useful values to stdout for the first two epochs only.
 * For debug/study purpose.
 *********************************************************************/
void dumpNetworkValues(int epoch)
{
    int w = 0;

    if (epoch < 2) {
        printf("--- Epoch %d ---\n", epoch);
        printf("SSE  = %.4lf\n", errorSSE);
        printf("LSE  = %.4lf\n", errorLSE);
        printf("MSE  = %.4lf\n", errorMSE);
        printf("RMSE = %.4lf\n", errorRMSE);
        printf("----------------\n");

        for (w = 0; w < NUM_OF_WEIGHTS; ++w) {
            printf("w[%d] = %.17lf\n", w, weight[w].value);
        }
        printf("\n");
    }
}

