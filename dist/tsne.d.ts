import * as tf from '@tensorflow/tfjs-core';
export interface TSNEConfiguration {
    perplexity?: number;
    exaggeration?: number;
    exaggerationIter?: number;
    exaggerationDecayIter?: number;
    momentum?: number;
    verbose?: boolean;
    knnMode: 'auto' | 'bruteForce';
}
export declare function maximumPerplexity(): number;
export declare function tsne(data: tf.Tensor, config?: TSNEConfiguration): TSNE;
export declare class TSNE {
    private data;
    private numPoints;
    private numDimensions;
    private numNeighbors;
    private packedData;
    private verbose;
    private knnEstimator;
    private optimizer;
    private config;
    private initialized;
    private probabilitiesInitialized;
    private knnMode;
    constructor(data: tf.Tensor, config?: TSNEConfiguration);
    private initialize();
    compute(iterations?: number): Promise<void>;
    iterateKnn(iterations?: number): Promise<void>;
    iterate(iterations?: number): Promise<void>;
    knnIterations(): number;
    coordinates(normalized?: boolean): tf.Tensor;
    coordsArray(normalized?: boolean): Promise<number[][]>;
    knnTotalDistance(): Promise<number>;
    private initializeProbabilities();
}
