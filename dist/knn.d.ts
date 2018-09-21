import * as tf from '@tensorflow/tfjs-core';
import { RearrangedData } from './interfaces';
export interface CustomDataDefinition {
    distanceComputationCode: string;
}
export declare class KNNEstimator {
    private verbose;
    private backend;
    private gpgpu;
    private _iteration;
    private numNeighs;
    private bruteForceKNNProgram;
    private randomSamplingKNNProgram;
    private kNNDescentProgram;
    private copyDistancesProgram;
    private copyIndicesProgram;
    private linesVertexIdBuffer;
    private dataTexture;
    private knnTexture0;
    private knnTexture1;
    private knnDataShape;
    readonly knnShape: RearrangedData;
    readonly iteration: number;
    readonly pointsPerIteration: number;
    constructor(dataTexture: WebGLTexture, dataFormat: RearrangedData | CustomDataDefinition, numPoints: number, numDimensions: number, numNeighs: number, verbose?: boolean);
    private log(str, obj?);
    private initializeTextures();
    private initilizeCustomWebGLPrograms(distanceComputationSource);
    iterateBruteForce(): void;
    iterateRandomSampling(): void;
    iterateKNNDescent(): void;
    knn(): WebGLTexture;
    distancesTensor(): tf.Tensor;
    indicesTensor(): tf.Tensor;
    forceFlush(): void;
    private downloadTextureToMatrix(texture);
    private iterateGPU(dataTexture, _iteration, startingKNNTexture, targetTexture?);
    private iterateRandomSamplingGPU(dataTexture, _iteration, startingKNNTexture, targetTexture?);
    private iterateKNNDescentGPU(dataTexture, _iteration, startingKNNTexture, targetTexture?);
}
