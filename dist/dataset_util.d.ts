import { RearrangedData } from './interfaces';
export declare function generateDistanceComputationSource(format: RearrangedData): string;
export declare function generateMNISTDistanceComputationSource(): string;
export declare function generateKNNClusterTexture(numPoints: number, numClusters: number, numNeighbors: number): {
    knnGraph: WebGLTexture;
    dataShape: RearrangedData;
};
export declare function generateKNNLineTexture(numPoints: number, numNeighbors: number): {
    knnGraph: WebGLTexture;
    dataShape: RearrangedData;
};
export declare function generateKNNClusterData(numPoints: number, numClusters: number, numNeighbors: number): {
    distances: Float32Array;
    indices: Uint32Array;
};
export declare function generateKNNLineData(numPoints: number, numNeighbors: number): {
    distances: Float32Array;
    indices: Uint32Array;
};
