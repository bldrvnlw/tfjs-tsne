"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var gl_util = require("./gl_util");
function generateDistanceComputationSource(format) {
    var source = "\n    #define DATA_NUM_PACKED_DIMENSIONS " + format.pixelsPerPoint + ".\n    #define DATA_POINTS_PER_ROW " + format.pointsPerRow + ".\n    #define DATA_NUM_ROWS " + format.numRows + ".\n    #define TEXTURE_WIDTH " + format.pointsPerRow * format.pixelsPerPoint + ".\n\n    //returns the texture coordinate for point/dimension\n    vec2 dataTexCoordinates(int id, int dimension) {\n      float id_f = float(id);\n      float row = (floor(id_f/DATA_POINTS_PER_ROW)+0.5) / DATA_NUM_ROWS;\n      float col = ((mod(id_f,DATA_POINTS_PER_ROW)*(DATA_NUM_PACKED_DIMENSIONS)\n                  + float(dimension)) + 0.5) / (TEXTURE_WIDTH);\n      return vec2(col,row);\n    }\n\n    //compute the euclidean squared distances between two points i and j\n    float pointDistanceSquared(int i, int j) {\n      vec4 result = vec4(0,0,0,0);\n      int num_iter = int(DATA_NUM_PACKED_DIMENSIONS);\n      for(int d = 0; d < num_iter; ++d) {\n        vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n        vec4 vj = texture(data_tex,dataTexCoordinates(j,d));\n        result += (vi-vj)*(vi-vj);\n      }\n      return (result.r+result.g+result.b+result.a);\n    }\n\n    //compute the euclidean squared distances between two points i and j\n    vec4 pointDistanceSquaredBatch(int i, int j0, int j1, int j2, int j3) {\n      vec4 result = vec4(0,0,0,0);\n      int num_iter = int(DATA_NUM_PACKED_DIMENSIONS);\n      for(int d = 0; d < num_iter; ++d) {\n        vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n        vec4 vj0 = texture(data_tex,dataTexCoordinates(j0,d));\n        vec4 vj1 = texture(data_tex,dataTexCoordinates(j1,d));\n        vec4 vj2 = texture(data_tex,dataTexCoordinates(j2,d));\n        vec4 vj3 = texture(data_tex,dataTexCoordinates(j3,d));\n        vj0 = (vi-vj0); vj0 *= vj0;\n        vj1 = (vi-vj1); vj1 *= vj1;\n        vj2 = (vi-vj2); vj2 *= vj2;\n        vj3 = (vi-vj3); vj3 *= vj3;\n        result.r += (vj0.r+vj0.g+vj0.b+vj0.a);\n        result.g += (vj1.r+vj1.g+vj1.b+vj1.a);\n        result.b += (vj2.r+vj2.g+vj2.b+vj2.a);\n        result.a += (vj3.r+vj3.g+vj3.b+vj3.a);\n      }\n      return result;\n    }\n    ";
    return source;
}
exports.generateDistanceComputationSource = generateDistanceComputationSource;
function generateMNISTDistanceComputationSource() {
    var source = "\n  #define POINTS_PER_ROW 250.\n  #define NUM_ROWS 240.\n  #define TEXTURE_WIDTH 3500.\n  #define TEXTURE_HEIGHT 3360.\n  #define DIGIT_WIDTH 14.\n  #define NUM_PACKED_DIMENSIONS 196\n\n  //returns the texture coordinate for point/dimension\n  vec2 dataTexCoordinates(int id, int dimension) {\n    float id_f = float(id);\n    float dimension_f = float(dimension);\n    float col = ((mod(id_f,POINTS_PER_ROW)*DIGIT_WIDTH));\n    float row = (floor(id_f/POINTS_PER_ROW)*DIGIT_WIDTH);\n\n    return (vec2(col,row)+\n            vec2(mod(dimension_f,DIGIT_WIDTH),floor(dimension_f/DIGIT_WIDTH))+\n            vec2(0.5,0.5)\n            )/\n            vec2(TEXTURE_WIDTH,TEXTURE_HEIGHT);\n  }\n\n  //compute the euclidean squared distances between two points i and j\n  float pointDistanceSquared(int i, int j) {\n    vec4 result = vec4(0,0,0,0);\n    for(int d = 0; d < NUM_PACKED_DIMENSIONS; d+=1) {\n      vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n      vec4 vj = texture(data_tex,dataTexCoordinates(j,d));\n      result += (vi-vj)*(vi-vj);\n    }\n    return (result.r+result.g+result.b+result.a);\n  }\n\n  //compute the euclidean squared distances between two points i and j\n  vec4 pointDistanceSquaredBatch(int i, int j0, int j1, int j2, int j3) {\n    vec4 result = vec4(0,0,0,0);\n    for(int d = 0; d < NUM_PACKED_DIMENSIONS; d+=1) {\n      vec4 vi = texture(data_tex,dataTexCoordinates(i,d));\n      vec4 vj0 = texture(data_tex,dataTexCoordinates(j0,d));\n      vec4 vj1 = texture(data_tex,dataTexCoordinates(j1,d));\n      vec4 vj2 = texture(data_tex,dataTexCoordinates(j2,d));\n      vec4 vj3 = texture(data_tex,dataTexCoordinates(j3,d));\n      vj0 = (vi-vj0); vj0 *= vj0;\n      vj1 = (vi-vj1); vj1 *= vj1;\n      vj2 = (vi-vj2); vj2 *= vj2;\n      vj3 = (vi-vj3); vj3 *= vj3;\n      result.r += (vj0.r+vj0.g+vj0.b+vj0.a);\n      result.g += (vj1.r+vj1.g+vj1.b+vj1.a);\n      result.b += (vj2.r+vj2.g+vj2.b+vj2.a);\n      result.a += (vj3.r+vj3.g+vj3.b+vj3.a);\n    }\n    return result;\n  }\n  ";
    return source;
}
exports.generateMNISTDistanceComputationSource = generateMNISTDistanceComputationSource;
function generateKNNClusterTexture(numPoints, numClusters, numNeighbors) {
    var pointsPerRow = Math.max(1, Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors));
    var numRows = Math.ceil(numPoints / pointsPerRow);
    var dataShape = { numPoints: numPoints, pixelsPerPoint: numNeighbors, numRows: numRows, pointsPerRow: pointsPerRow };
    var pointsPerCluster = Math.ceil(numPoints / numClusters);
    var textureValues = new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
    for (var i = 0; i < numPoints; ++i) {
        var clusterId = Math.floor(i / pointsPerCluster);
        for (var n = 0; n < numNeighbors; ++n) {
            var id = (i * numNeighbors + n) * 2;
            textureValues[id] = Math.floor(Math.random() * pointsPerCluster) +
                clusterId * pointsPerCluster;
            textureValues[id + 1] = Math.random();
        }
    }
    var backend = tf.ENV.findBackend('webgl');
    if (backend === null) {
        throw Error('WebGL backend is not available');
    }
    var gpgpu = backend.getGPGPUContext();
    var knnGraph = gl_util.createAndConfigureTexture(gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);
    return { knnGraph: knnGraph, dataShape: dataShape };
}
exports.generateKNNClusterTexture = generateKNNClusterTexture;
function generateKNNLineTexture(numPoints, numNeighbors) {
    var pointsPerRow = Math.max(1, Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors));
    var numRows = Math.ceil(numPoints / pointsPerRow);
    var dataShape = { numPoints: numPoints, pixelsPerPoint: numNeighbors, numRows: numRows, pointsPerRow: pointsPerRow };
    var textureValues = new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
    for (var i = 0; i < numPoints; ++i) {
        for (var n = 0; n < numNeighbors; ++n) {
            var id = (i * numNeighbors + n) * 2;
            textureValues[id] =
                Math.floor(i + n - (numNeighbors / 2) + numPoints) % numPoints;
            textureValues[id + 1] = 1;
        }
    }
    var backend = tf.ENV.findBackend('webgl');
    if (backend === null) {
        throw Error('WebGL backend is not available');
    }
    var gpgpu = backend.getGPGPUContext();
    var knnGraph = gl_util.createAndConfigureTexture(gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);
    return { knnGraph: knnGraph, dataShape: dataShape };
}
exports.generateKNNLineTexture = generateKNNLineTexture;
function generateKNNClusterData(numPoints, numClusters, numNeighbors) {
    var pointsPerCluster = Math.ceil(numPoints / numClusters);
    var distances = new Float32Array(numPoints * numNeighbors);
    var indices = new Uint32Array(numPoints * numNeighbors);
    for (var i = 0; i < numPoints; ++i) {
        var clusterId = Math.floor(i / pointsPerCluster);
        for (var n = 0; n < numNeighbors; ++n) {
            var id = (i * numNeighbors + n);
            distances[id] = Math.random();
            indices[id] = Math.floor(Math.random() * pointsPerCluster) +
                clusterId * pointsPerCluster;
        }
    }
    return { distances: distances, indices: indices };
}
exports.generateKNNClusterData = generateKNNClusterData;
function generateKNNLineData(numPoints, numNeighbors) {
    var distances = new Float32Array(numPoints * numNeighbors);
    var indices = new Uint32Array(numPoints * numNeighbors);
    for (var i = 0; i < numPoints; ++i) {
        for (var n = 0; n < numNeighbors; ++n) {
            var id = (i * numNeighbors + n);
            distances[id] = 1;
            indices[id] =
                Math.floor(i + n - (numNeighbors / 2) + numPoints) % numPoints;
        }
    }
    return { distances: distances, indices: indices };
}
exports.generateKNNLineData = generateKNNLineData;
//# sourceMappingURL=dataset_util.js.map