//
//  main.c
//  NeuralNet
//
//  Created by Parker on 11/1/16.
//  Copyright © 2016 Parker. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double rrand() {
    return (double)rand() / (double)RAND_MAX;
//    static double inc=0;
//    return (inc++)/8;
}
double sigmoid(double x) {
    return 1/(1+pow(M_E,-x));
//    return x;
}
double sigmoidinv(double x) {
    return pow(M_E,-x)/pow((1+pow(M_E,-x)),2);
//    return 1;
}
typedef struct {
    double learning_rate;
    int numinputs;
    int numlayers;
    int* layer_sizes;
    double* backpropc1;
    double* backpropc2;
    
    double* inputs;
    double** layers;
    double** invsigmoid;
    double** weights;
    double** deltaweights;
} NeuralNet;

NeuralNet makeNeuralNet(int numhiddenlayers,int numinputs,int* layersizes) {
    NeuralNet target;
    target.numlayers    = numhiddenlayers+1;
    target.layer_sizes  = layersizes;
    target.numinputs    = numinputs;
    target.inputs       = malloc(sizeof(double)*numinputs);
    target.layers       = malloc(sizeof(double*)*target.numlayers);
    target.invsigmoid   = malloc(sizeof(double*)*target.numlayers);
    target.weights      = malloc(sizeof(double*)*target.numlayers);
    target.deltaweights = malloc(sizeof(double*)*target.numlayers);
    int lastlayer=numinputs;
    for (int layer=0;layer<target.numlayers;layer++) {
        target.layers[layer]       = malloc(sizeof(double)*layersizes[layer]);
        target.invsigmoid[layer]   = malloc(sizeof(double)*layersizes[layer]);
        target.weights[layer]      = malloc(sizeof(double)*lastlayer*layersizes[layer]);
        target.deltaweights[layer] = malloc(sizeof(double)*lastlayer*layersizes[layer]);
        for (int init=0;init<lastlayer*layersizes[layer];init++) {
            target.weights[layer][init]=rrand();
        }
        lastlayer=layersizes[layer];
    }
    int maxlayersize=0;
    for (int layer=0;layer<target.numlayers-1;layer++) {
        if (layersizes[layer]>maxlayersize) {
            maxlayersize=layersizes[layer];
        }
    }
//    printf("%d\n",maxlayersize);
    target.backpropc1 = malloc(sizeof(double)*maxlayersize);
    target.backpropc2 = malloc(sizeof(double)*maxlayersize);
    return target;
}
void representweights(NeuralNet* target) {
    int prevlen=target->numinputs;
    for (int layer=0;layer<target->numlayers;layer++) {
        printf("=========\n");
        for (int y=0;y<target->layer_sizes[layer];y++) {
            printf("[");
            for (int x=0;x<prevlen;x++) {
                printf("%f  ",target->weights[layer][x+prevlen*y]);
            }
            printf("]\n");
        }
        prevlen=target->layer_sizes[layer];
    }
    printf("=========\n");
}
void representdeltaweights(NeuralNet* target) {
    int prevlen=target->numinputs;
    for (int layer=0;layer<target->numlayers;layer++) {
        printf("=========\n");
        for (int y=0;y<target->layer_sizes[layer];y++) {
            printf("[");
            for (int x=0;x<prevlen;x++) {
                printf("%f  ",target->deltaweights[layer][x+prevlen*y]);
            }
            printf("]\n");
        }
        prevlen=target->layer_sizes[layer];
    }
    printf("=========\n");
}
void representnodes(NeuralNet* target) {
    printf("=========\n");
    for (int layer=0;layer<target->numlayers;layer++) {
        printf("[");
        for (int x=0;x<target->layer_sizes[layer];x++) {
            printf("%f  ",target->layers[layer][x]);
        }
        printf("]\n");
    }
    printf("=========\n");
}
void cleanup(NeuralNet* target) {
    for (int layer=0;layer<target->numlayers;layer++) {
        free(target->layers[layer]);
        free(target->invsigmoid[layer]);
        free(target->weights[layer]);
        free(target->deltaweights[layer]);
    }
    free(target->backpropc1);
    free(target->backpropc2);
    free(target->layers);
    free(target->weights);
    free(target->invsigmoid);
    free(target->layer_sizes);
    free(target->deltaweights);
    free(target->inputs);
}
double* propogate(NeuralNet* target) {
    int xsize = target->numinputs;
    double* prevdataset = target->inputs;
    
    for (int layer=0;layer<target->numlayers;layer++) {
        for (int y=0;y<target->layer_sizes[layer];y++) {
            double sum=0.0;
            for (int x=0;x<xsize;x++) {
                sum+=prevdataset[x]*target->weights[layer][x+xsize*y];
            }
            target->layers[layer][y]  = sigmoid(sum);
            target->invsigmoid[layer][y] = sigmoidinv(sum);
        }
        prevdataset = target->layers[layer];
        xsize = target->layer_sizes[layer];
    }
    return target->layers[target->numlayers-1];
}
void backpropogate(NeuralNet* target,double* goals) {
    int prevlen=target->numinputs;
    for (int layer=0;layer<target->numlayers;layer++) {
        for (int x=0;x<prevlen;x++) {
            for (int y=0;y<target->layer_sizes[layer];y++) {
                target->deltaweights[layer][x+prevlen*y]=0;
            }
        }
        prevlen=target->layer_sizes[layer];
    }
    for (int goal=0;goal<target->layer_sizes[target->numlayers-1];goal++) {
        double diff = goals[goal]-target->layers[target->numlayers-1][goal];
//        printf("%f is the difference \n",diff);


//        double* incoming = target->backpropc1;
//        double* incoming = malloc(sizeof(double));
        target->backpropc1[0] = target->invsigmoid[target->numlayers-1][goal];
        int incominglen=1;
        
        for (int layer=target->numlayers-1;layer>=0;layer--) {
            int prevlen;
            double* prevlist;
            if (layer==0) {
                prevlen=target->numinputs;
                prevlist=target->inputs;
            } else {
                prevlen=target->layer_sizes[layer-1];
                prevlist=target->layers[layer-1];
            }
            
            for (int x=0;x<prevlen;x++) {
                for (int y=0;y<incominglen;y++) {
                    target->deltaweights[layer][x+prevlen*y]+=prevlist[x]*target->backpropc1[y]*diff;
                }
            }
            if (layer!=0) {
//                double* nextcoming = malloc(sizeof(double)*prevlen);
                double* nextcoming = target->backpropc2;
                for (int x=0;x<prevlen;x++) {
                    double sum=0.0;
                    for (int y=0;y<incominglen;y++) {
                        sum+=target->weights[layer][x+prevlen*y]*target->backpropc1[y];
                    }
                    nextcoming[x]=sum;
                }
//                free(incoming);
//                incoming=nextcoming;
                target->backpropc2 = target->backpropc1;
                target->backpropc1 = nextcoming;
                
                incominglen=prevlen;
                for (int x=0;x<incominglen;x++) {
                    target->backpropc1[x]*=target->invsigmoid[layer-1][x];
                }
            }
        }
//        free(incoming);
    }
    int prevlen1=target->numinputs;
    for (int layer=0;layer<target->numlayers;layer++) {
        for (int x=0;x<prevlen1;x++) {
            for (int y=0;y<target->layer_sizes[layer];y++) {
                target->weights[layer][x+prevlen*y]+=target->deltaweights[layer][x+prevlen*y]*target->learning_rate;
            }
        }
        prevlen1=target->layer_sizes[layer];
    }
}
int main(int argc, const char * argv[]) {
//    double* weights = malloc(sizeof(double)*());
//    double* nodes = malloc(sizeof(double)*(2+3+1));
    int numinputs = 2;
    int numhiddenlayers = 1;
    int numoutputs = 1;
    int* hidden_layer_sizes = malloc(sizeof(int)*(numhiddenlayers+1));
    hidden_layer_sizes[0]=3;
    hidden_layer_sizes[numhiddenlayers]=numoutputs;
    
    NeuralNet myfamily = makeNeuralNet(numhiddenlayers, numinputs, hidden_layer_sizes);
    myfamily.learning_rate = .8;
    representweights(&myfamily);
    representnodes(&myfamily);
    
    
//    double* inputs = malloc(sizeof(double)*2);
//    inputs[0]=1;
//    inputs[1]=1;
//    printf("PROPOGATING: (1,1)\n");
//    propogate(&myfamily,inputs);
//    representweights(&myfamily);
//    representnodes(&myfamily);
//    double goal = 0;
//    backpropogate(&myfamily, &goal);
//    representdeltaweights(&myfamily);
//    
//    return 0;
    
    int epoch=0;
    int shouldnotbreak=1;
    double lasterror=0.0;
    while (shouldnotbreak) {
        shouldnotbreak=0;
        double totalerror=0.0;
        for (int x=0;x<2;x++) {
            for (int y=0;y<2;y++) {
                myfamily.inputs[0]=(double)x;
                myfamily.inputs[1]=(double)y;
                double result;
                result = *propogate(&myfamily);
                double goal=(double)!x;
                if (fabs(result-goal)>.01) {
                    shouldnotbreak=1;
                }
                backpropogate(&myfamily, &goal);
                totalerror+=fabs(result-goal);
            }
        }
        totalerror/=4.0;
        if (epoch%10000==0) {
            printf("round %d avg error %f delta avg error %f\n",epoch,totalerror,totalerror-lasterror);
        }
        lasterror=totalerror;
        epoch++;
    }
    printf("DONE\n");
    
    
    for (int x=0;x<2;x++) {
        for (int y=0;y<2;y++) {
            myfamily.inputs[0]=(double)x;
            myfamily.inputs[1]=(double)y;
            double result;
            result = *propogate(&myfamily);
            printf("INPUTS: %f, %f.  OUTPUT: %f\n",myfamily.inputs[0],myfamily.inputs[1],result);
        }
    }
    
    
    cleanup(&myfamily);
//    printf("%f",weights[0]);
//    printf("Hello, World!\n");
    return 0;
}














