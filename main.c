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
#include <setjmp.h>

static jmp_buf JMPBUF;

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.pch"


void custom_assert(int condition,const char* errormessage) {
    if (!condition) {
        printf("%s\n",errormessage);
    
        setjmp(JMPBUF);
    }
}

double rrand() {
//    return 1.0;
    return (double)rand() / (double)RAND_MAX;
//    static double inc=0;
//    return (inc++)/8;
}
double sigmoid(double x) {
//    return 
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
    
    double* inputs;
    double** layers;
    double** invsigmoid;
    double** weights;
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
    int lastlayer=numinputs;
    for (int layer=0;layer<target.numlayers;layer++) {
        target.layers[layer]       = malloc(sizeof(double)*layersizes[layer]);
        target.invsigmoid[layer]   = malloc(sizeof(double)*layersizes[layer]);
        target.weights[layer]      = malloc(sizeof(double)*lastlayer*layersizes[layer]);
        for (int init=0;init<lastlayer*layersizes[layer];init++) {
            target.weights[layer][init]=rrand();
        }
        lastlayer=layersizes[layer];
    }
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
void cleanup_neuralnet(NeuralNet* target) {
    for (int layer=0;layer<target->numlayers;layer++) {
        free(target->layers[layer]);
        free(target->invsigmoid[layer]);
        free(target->weights[layer]);
    }
    free(target->layers);
    free(target->weights);
    free(target->invsigmoid);
    free(target->layer_sizes);
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
    double* carriage=target->layers[target->numlayers-1];
    for (int goal=0;goal<target->layer_sizes[target->numlayers-1];goal++) {
        double diff = goals[goal]-target->layers[target->numlayers-1][goal];
        carriage[goal] = target->invsigmoid[target->numlayers-1][goal]*diff;
    }
    int incominglen=target->layer_sizes[target->numlayers-1];
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
            double sum=0.0;
            for (int y=0;y<incominglen;y++) {
                sum+=target->weights[layer][x+prevlen*y]*carriage[y];
                target->weights[layer][x+prevlen*y]+=prevlist[x]*carriage[y]*target->learning_rate;
            }
            prevlist[x]=sum;
        }
        carriage = prevlist;
        incominglen=prevlen;
        if (layer!=0) {
            for (int x=0;x<incominglen;x++) {
                carriage[x]*=target->invsigmoid[layer-1][x];
            }
        }
    }
}

void basic_example() {
    int numinputs = 2;
    int numhiddenlayers = 1;
    int numoutputs = 1;
    int* hidden_layer_sizes = malloc(sizeof(int)*(numhiddenlayers+1));
    hidden_layer_sizes[0]=3;
    hidden_layer_sizes[numhiddenlayers]=numoutputs;
    
    NeuralNet myfamily = makeNeuralNet(numhiddenlayers, numinputs, hidden_layer_sizes);
    myfamily.learning_rate = .8;
    
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
                double goal=(double)(x^y);
                if (fabs(result-goal)>.01) {
                    shouldnotbreak=1;
                }
                backpropogate(&myfamily, &goal);
                totalerror+=fabs(result-goal);
            }
        }
        totalerror/=4.0;
        if (epoch%100000==0) {
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
    
    cleanup_neuralnet(&myfamily);
}


typedef struct {
    int x;
    int y;
    unsigned char* data;
} Image;

Image load_image(const char* filename) {
    int k;
    Image result;
    printf("%s is being loaded\n",filename);
    result.data = stbi_load(filename, &result.x, &result.y, &k, 3);
    custom_assert(result.data!=0, "image loading failure");
    custom_assert(k==3, "image in wrong format: incorrect number of channels.");
    return result;
}
void cleanup_image(Image* image) {
    if (image!=0) {
        stbi_image_free(image->data);
    }
}

typedef struct {
    Image* input;
    double** kernel_conv_weights;
    double** layerbuffers;
    double* rgbconvolution;
    double* sourcebuffer;
    double* deltabuffer;
    int layer_pattern;
    int layers;
    int parallels;
    int inputsize;
    int outputsize;
    int kradius;
    int upscale;
    NeuralNet fully_connected_layer;
} ConvNeuralNet;

ConvNeuralNet makeConvNeuralNet(int parallels,int layer_pattern,int layers,int inputsize,int kradius,int fclayers,int* fclayersizes,int upscale) {
    ConvNeuralNet construct;
    construct.input = 0;
    construct.parallels     = parallels;
    construct.layer_pattern = layer_pattern;
    construct.layers        = layers;
    construct.inputsize     = inputsize;
    construct.kradius       = kradius;
    construct.upscale       = upscale;
    int width      = kradius*2+1;
    int convlayers = 0;
    for (int layer=0;layer<layers;layer++) {
        if (!(layer_pattern&(1<<layer))) {
            convlayers++;
        }
    }
    inputsize-=kradius<<1;
    construct.layerbuffers        = malloc(sizeof(double*)*(layers-1));
    construct.kernel_conv_weights = malloc(sizeof(double*)*convlayers);
    construct.sourcebuffer        = malloc(sizeof(double)*inputsize*inputsize*parallels);
    construct.rgbconvolution      = malloc(sizeof(double)*parallels*width*width*3);
    construct.deltabuffer         = calloc(width*width,sizeof(double));
    for (int para=0;para<parallels;para++) {
        for (int channel=0;channel<3;channel++) {
            for (int x=0;x<kradius*2+1;x++) {
                for (int y=0;y<kradius*2+1;y++) {
                    construct.rgbconvolution[para*3*width*width+channel*width*width+x*width+y]=rrand();
                }
            }
        }
    }
    int convlayer = 0;
    for (int layer=0;layer<layers;layer++) {
        if (layer_pattern&1) {
            custom_assert((inputsize&1)==0,"pooling layer must have divisible input layer");
            inputsize = inputsize>>1;
        } else {
            construct.kernel_conv_weights[convlayer]=malloc(sizeof(double)*width*width*parallels);
            for (int para=0;para<parallels;para++) {
                for (int x=0;x<width;x++) {
                    for (int y=0;y<width;y++) {
                        construct.kernel_conv_weights[convlayer][para*width*width+x*width+y]=rrand();
                    }
                }
            }
            inputsize -= kradius<<1;
            custom_assert(inputsize>0,"convolutional layer must have input layer at least the size of the kradius");
            convlayer++;
        }
        if (layer!=layers-1) {
            construct.layerbuffers[layer]=malloc(sizeof(double)*inputsize*inputsize*parallels);
        }
        layer_pattern = layer_pattern>>1;
    }
    construct.outputsize=inputsize;
    printf("%d is compressed size.\n",inputsize);
    construct.fully_connected_layer = makeNeuralNet(fclayers,inputsize*inputsize*parallels,fclayersizes);
    return construct;
}
void cleanup_convneuralnet(ConvNeuralNet* target) {
    cleanup_image(target->input);
    int convlayer=0;
    for (int layer=0;layer<target->layers;layer++) {
        if ((target->layer_pattern&(1<<layer))==0) {
            free(target->kernel_conv_weights[convlayer++]);
        }
        free(target->layerbuffers[layer]);
    }
    
    free(target->kernel_conv_weights);
    free(target->deltabuffer);
    free(target->layerbuffers);
    free(target->rgbconvolution);
    free(target->sourcebuffer);
    cleanup_neuralnet(&target->fully_connected_layer);
}

double* propogate_conv(ConvNeuralNet* target) {
    int kwidth=target->kradius*2+1;
    for (int para=0;para<target->parallels;para++) {
        int inputsize=target->inputsize;
        inputsize-=target->kradius<<1;
        for (int x=0;x<inputsize;x++) {
            for (int y=0;y<inputsize;y++) {
                double sum=0.0;
                for (int chan=0;chan<3;chan++) {
                    for (int x2=0;x2<kwidth;x2++) {
                        for (int y2=0;y2<kwidth;y2++) {
                            if (((x+x2)<<target->upscale)<target->input->x&&((y+y2)<<target->upscale)<target->input->y) {
                                int source = target->input->data[(((x+x2)<<target->upscale)*target->input->y+((y+y2)<<target->upscale))*3+chan];
                                source = (source<0)?source+256:source;
                                sum += target->rgbconvolution[para*kwidth*kwidth*3+chan*kwidth*kwidth+x2*kwidth+y2]*(((double)source)/255.0);
                            }
                        }
                    }
                }
                target->sourcebuffer[para*inputsize*inputsize+x*inputsize+y]=sum;
            }
        }
        double* prevbuf = target->sourcebuffer;
        int convlayer=0;
        for (int layer=0;layer<target->layers;layer++) {
            int width = inputsize;
            if (target->layer_pattern&(1<<layer)) {
                inputsize = inputsize>>1;
                for (int x=0;x<inputsize;x++) {
                    for (int y=0;y<inputsize;y++) {
                        double m1=prevbuf[para*width*width+(x<<1)    *width+(y<<1)];
                        double m2=prevbuf[para*width*width+((x<<1)+1)*width+(y<<1)];
                        double m3=prevbuf[para*width*width+(x<<1)    *width+(y<<1)+1];
                        double m4=prevbuf[para*width*width+((x<<1)+1)*width+(y<<1)+1];
                        double sum =
                        ((m1=
                        ((m1=
                        (m1
                        >m2)?m1:m2)
                        >m3)?m1:m3)
                        >m4)?m1:m4;
                        if (layer!=target->layers-1) {
                            target->layerbuffers[layer][para*inputsize*inputsize+x*inputsize+y]=sum;
                        } else {
                            target->fully_connected_layer.inputs[para*inputsize*inputsize+x*inputsize+y]=sum;
                        }
                    }
                }
            } else {
                inputsize -= target->kradius<<1;
                for (int x=0;x<inputsize;x++) {
                    for (int y=0;y<inputsize;y++) {
                        double sum=0.0;
                        for (int x2=0;x2<kwidth;x2++) {
                            for (int y2=0;y2<kwidth;y2++) {
                                sum += prevbuf[para*width*width+(x+x2)*width+y+y2]*target->kernel_conv_weights[convlayer][para*kwidth*kwidth+x2*kwidth+y2];
                            }
                        }
                        if (layer!=target->layers-1) {
                            target->layerbuffers[layer][para*inputsize*inputsize+x*inputsize+y]=sum;
                        } else {
                            target->fully_connected_layer.inputs[para*inputsize*inputsize+x*inputsize+y]=sum;
                        }
                    }
                }
                convlayer++;
            }
            prevbuf=target->layerbuffers[layer];
        }
    }
    return propogate(&target->fully_connected_layer);
}
void backpropogate_conv(ConvNeuralNet* target,double* goals) {
    int convlayer=-1;
    for (int layer=0;layer<target->layers;layer++) {
        if ((target->layer_pattern&(1<<layer))==0) {
            convlayer++;
        }
    }
    backpropogate(&target->fully_connected_layer,goals);
    double* carriage = target->fully_connected_layer.inputs;
    double* destination;
    int revinputsize=target->outputsize;
    
    int kwidth=target->kradius*2+1;
    for (int layer=target->layers-1;layer>=0;layer--) {
        if (layer==0) {
            destination = target->sourcebuffer;
        } else {
            destination = target->layerbuffers[layer-1];
        }
        if (target->layer_pattern&(1<<layer)) {
            int width=revinputsize;
            revinputsize=revinputsize<<1;
            for (int para=0;para<target->parallels;para++) {
                for (int x=0;x<width;x++) {
                    for (int y=0;y<width;y++) {
                        double m1=destination[para*revinputsize*revinputsize+(x<<1)    *revinputsize+(y<<1)];
                        double m2=destination[para*revinputsize*revinputsize+((x<<1)+1)*revinputsize+(y<<1)];
                        double m3=destination[para*revinputsize*revinputsize+(x<<1)    *revinputsize+(y<<1)+1];
                        double m4=destination[para*revinputsize*revinputsize+((x<<1)+1)*revinputsize+(y<<1)+1];
                        destination[para*revinputsize*revinputsize+(x<<1)    *revinputsize+(y<<1)]=0;
                        destination[para*revinputsize*revinputsize+((x<<1)+1)*revinputsize+(y<<1)]=0;
                        destination[para*revinputsize*revinputsize+(x<<1)    *revinputsize+(y<<1)+1]=0;
                        destination[para*revinputsize*revinputsize+((x<<1)+1)*revinputsize+(y<<1)+1]=0;
                        if (m1>m2&&m1>m3&&m1>m4) {
                            destination[para*revinputsize*revinputsize+(x<<1)    *revinputsize+(y<<1)]=carriage[para*width*width+x*width+y];
                        } else if (m2>m3&&m2>m4) {
                            destination[para*revinputsize*revinputsize+((x<<1)+1)*revinputsize+(y<<1)]=carriage[para*width*width+x*width+y];
                        } else if (m3>m4) {
                            destination[para*revinputsize*revinputsize+(x<<1)    *revinputsize+(y<<1)+1]=carriage[para*width*width+x*width+y];
                        } else {
                            destination[para*revinputsize*revinputsize+((x<<1)+1)*revinputsize+(y<<1)+1]=carriage[para*width*width+x*width+y];
                        }
                    }
                }
            }
                //pooling
        } else {
            int width=revinputsize;
            revinputsize=revinputsize+(target->kradius<<1);
            for (int para=0;para<target->parallels;para++) {
                for (int x=0;x<width;x++) {
                    for (int y=0;y<width;y++) {
                        for (int x2=0;x2<kwidth;x2++) {
                            for (int y2=0;y2<kwidth;y2++) {
                                target->deltabuffer[x2*kwidth+y2]+=
                                destination[para*revinputsize*revinputsize+(x+x2)*revinputsize+y+y2]*carriage[para*width*width+x*width+y];
                            }
                        }
                        
                    }
                }
                for (int x=0;x<revinputsize;x++) {
                    for (int y=0;y<revinputsize;y++) {
                        destination[para*revinputsize*revinputsize+x*revinputsize+y]=0;
                    }
                }
                for (int x=0;x<width;x++) {
                    for (int y=0;y<width;y++) {
                        for (int x2=0;x2<kwidth;x2++) {
                            for (int y2=0;y2<kwidth;y2++) {
                                destination[para*revinputsize*revinputsize+(x+x2)*revinputsize+y+y2]+=
                                carriage[para*width*width+x*width+y]*target->kernel_conv_weights[convlayer][para*kwidth*kwidth+x2*kwidth+y2];
                            }
                        }
                    }
                }
                for (int x2=0;x2<kwidth;x2++) {
                    for (int y2=0;y2<kwidth;y2++) {
                        target->kernel_conv_weights[convlayer][para*kwidth*kwidth+x2*kwidth+y2]+=
                        target->deltabuffer[x2*kwidth+y2]*target->fully_connected_layer.learning_rate;
                        target->deltabuffer[x2*kwidth+y2]=0;
                    }
                }
            }
            convlayer--;
        }
        carriage=destination;
    }
    for (int para=0;para<target->parallels;para++) {
        int inputsize=target->inputsize;
        inputsize-=target->kradius<<1;
        for (int chan=0;chan<3;chan++) {
        
            for (int x=0;x<inputsize;x++) {
                for (int y=0;y<inputsize;y++) {
                    double spixel = target->sourcebuffer[para*inputsize*inputsize+x*inputsize+y];
                
                    for (int x2=0;x2<kwidth;x2++) {
                        for (int y2=0;y2<kwidth;y2++) {
                            if ((x+x2)<target->input->x&&(y+y2)<target->input->y) {
                                int source = target->input->data[((x+x2)*target->input->y+y+y2)*3+chan];
                                source = (source<0)?source+256:source;
                                target->deltabuffer[x2*kwidth+y2]+= spixel*(((double)source)/255.0);
//                                sum += target->rgbconvolution[para*kwidth*kwidth*3+chan*kwidth*kwidth+x2*kwidth+y2]*(((double)source)/255.0);
                            }
                        }
                    }
                    
                }
            }
            for (int x2=0;x2<kwidth;x2++) {
                for (int y2=0;y2<kwidth;y2++) {
                    target->rgbconvolution[para*kwidth*kwidth*3+chan*kwidth*kwidth+x2*kwidth+y2]+=
//                    target->kernel_conv_weights[convlayer][para*kwidth*kwidth+x2*kwidth+y2]+=
                    target->deltabuffer[x2*kwidth+y2]*target->fully_connected_layer.learning_rate;
                    target->deltabuffer[x2*kwidth+y2]=0;
                }
            }
        }
    }
}
void convolutional_example() {
    Image* foximages = malloc(sizeof(Image)*248);
    int foximagenumber=0;
    for (int foxim=1;foxim<=248;foxim++) {
        char converted[100];
        sprintf(converted, "/Users/legalian/dev/NeuralNet/NeuralNet/redfox_trainingdata/pic_%03d.jpg", foxim);
        Image im = load_image(converted);
        if (im.x<=300&&im.y<=300) {
            foximages[foximagenumber++]=im;
        }
    }
    Image* notfoximages = malloc(sizeof(Image)*193);
    int notfoximagenumber=0;
    for (int foxim=1;foxim<=193;foxim++) {
        char converted[100];
        sprintf(converted, "/Users/legalian/dev/NeuralNet/NeuralNet/animal_notfox_trainingdata/pic_%03d.jpg", foxim);
        Image im = load_image(converted);
        if (im.x<=300&&im.y<=300) {
            notfoximages[notfoximagenumber++]=im;
        }
    }
    int numhiddenlayers = 1;
    int numoutputs = 1;
    int* hidden_layer_sizes = malloc(sizeof(int)*(numhiddenlayers+1));
    hidden_layer_sizes[0]=15;
    hidden_layer_sizes[numhiddenlayers]=numoutputs;
    ConvNeuralNet myfamily = makeConvNeuralNet(3,
    //assumed 0, 300->290
    (0<<0)+//280
    (1<<1)+//140
    (0<<2)+//130
    (0<<3)+//120
    (1<<4)+//60
    (0<<5)+//50
    (0<<6)+//40
    (1<<7)+//20
    (1<<8)+//10
    (1<<9)//5
    ,10,300,5,numhiddenlayers,hidden_layer_sizes,2);
    myfamily.fully_connected_layer.learning_rate = 1;
    
    
//        printf("%f\n",*propogate_conv(&myfamily));


//        int epoch=0;
    int foxestested=0;
    int nonfoxestested=0;
//        int shouldnotbreak=1;
    while (1) {
        double target;
        if (foxestested>nonfoxestested) {
            int imageindex=nonfoxestested%notfoximagenumber;
            myfamily.input=notfoximages+imageindex;
            double result;
            result = *propogate_conv(&myfamily);
            printf("NOT a fox, returned %f.\n",result);
            target=0;
            backpropogate_conv(&myfamily, &target);
            nonfoxestested++;
        } else {
            int imageindex=foxestested%foximagenumber;
            myfamily.input=foximages+imageindex;
            double result;
            result = *propogate_conv(&myfamily);
            printf("IS  a fox, returned %f.\n",result);
            target=1;
            backpropogate_conv(&myfamily, &target);
            foxestested++;
        }
    }
    printf("DONE\n");
    cleanup_convneuralnet(&myfamily);
}


int main(int argc, const char * argv[]) {
    if (setjmp(JMPBUF)) {
        return 1;
    } else {
        
        basic_example();
        
    }
    return 0;
}



