import * as tf from '@tensorflow/tfjs-core';
export interface RearrangedData {
    numPoints: number;
    pointsPerRow: number;
    pixelsPerPoint: number;
    numRows: number;
}
export declare function createBruteForceKNNProgram(gpgpu: tf.webgl.GPGPUContext, numNeighbors: number, distanceComputationSource: string): WebGLProgram;
export declare function createRandomSamplingKNNProgram(gpgpu: tf.webgl.GPGPUContext, numNeighbors: number, distanceComputationSource: string): WebGLProgram;
export declare function createKNNDescentProgram(gpgpu: tf.webgl.GPGPUContext, numNeighbors: number, distanceComputationSource: string): WebGLProgram;
export interface RearrangedData {
    numPoints: number;
    pointsPerRow: number;
    pixelsPerPoint: number;
    numRows: number;
}
export declare function executeKNNProgram(gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, dataTex: WebGLTexture, startingKNNTex: WebGLTexture, iteration: number, knnShape: RearrangedData, vertexIdBuffer: WebGLBuffer, targetTex?: WebGLTexture): void;
export declare function createCopyDistancesProgram(gpgpu: tf.webgl.GPGPUContext): WebGLProgram;
export declare function executeCopyDistancesProgram(gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, knnTex: WebGLTexture, knnShape: RearrangedData, targetTex?: WebGLTexture): void;
export declare function createCopyIndicesProgram(gpgpu: tf.webgl.GPGPUContext): WebGLProgram;
export declare function executeCopyIndicesProgram(gpgpu: tf.webgl.GPGPUContext, program: WebGLProgram, knnTex: WebGLTexture, knnShape: RearrangedData, targetTex?: WebGLTexture): void;
