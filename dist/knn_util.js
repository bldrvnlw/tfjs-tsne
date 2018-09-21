"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var gl_util = require("./gl_util");
function generateFragmentShaderSource(distanceComputationSource, numNeighbors) {
    var source = "#version 300 es\n    precision highp float;\n    uniform sampler2D data_tex;\n    uniform float num_points;\n    uniform float points_per_row_knn;\n    uniform float num_rows_knn;\n    uniform float num_neighs;\n    uniform float iteration;\n\n    #define NUM_PACKED_NEIGHBORS " + numNeighbors / 4 + "\n\n    flat in vec4 knn[NUM_PACKED_NEIGHBORS];\n    flat in int point_id;\n    in float neighbor_id;\n\n    const float MAX_DIST = 10e30;\n\n    " + distanceComputationSource + "\n\n    out vec4 fragmentColor;\n    void main() {\n      int id = int(neighbor_id/4.);\n      int channel = int(mod(neighbor_id,4.)+0.1);\n\n      if(channel == 0) {\n        fragmentColor = vec4(knn[id].r,0,0,1);\n      }else if(channel == 1) {\n        fragmentColor = vec4(knn[id].g,0,0,1);\n      }else if(channel == 2) {\n        fragmentColor = vec4(knn[id].b,0,0,1);\n      }else if(channel == 3) {\n        fragmentColor = vec4(knn[id].a,0,0,1);\n      }\n\n      //If the neighbor has a valid id i compute the distance squared\n      //otherwise I set it to invalid\n      if(fragmentColor.r >= 0.) {\n        fragmentColor.g = pointDistanceSquared(int(fragmentColor.r),point_id);\n      }else{\n        fragmentColor.g = MAX_DIST;\n      }\n    }\n  ";
    return source;
}
function generateVariablesAndDeclarationsSource(numNeighbors) {
    var source = "\n  precision highp float;\n  #define NEIGH_PER_ITER 20\n  #define NUM_NEIGHBORS " + numNeighbors + "\n  #define NUM_NEIGHBORS_FLOAT " + numNeighbors + ".\n  #define NUM_PACKED_NEIGHBORS " + numNeighbors / 4 + "\n  #define MAX_DIST 10e30\n\n  //attributes\n  in float vertex_id;\n  //uniforms\n  uniform sampler2D data_tex;\n  uniform sampler2D starting_knn_tex;\n  uniform float num_points;\n  uniform float points_per_row_knn;\n  uniform float num_rows_knn;\n  uniform float num_neighs;\n  uniform float iteration;\n\n  //output\n  //the indices are packed in varying vectors\n  flat out vec4 knn[NUM_PACKED_NEIGHBORS];\n  //used to recover the neighbor id in the fragment shader\n  out float neighbor_id;\n  //used to recover the point id in the fragment shader\n  //(for recomputing distances)\n  flat out int point_id;\n\n  float distances_heap[NUM_NEIGHBORS];\n  int knn_heap[NUM_NEIGHBORS];\n  ";
    return source;
}
var randomGeneratorSource = "\n//Random function developed by Inigo Quilez\n//https://www.shadertoy.com/view/llGSzw\nfloat hash1( uint n ) {\n    // integer hash copied from Hugo Elias\n\t  n = (n << 13U) ^ n;\n    n = n * (n * n * 15731U + 789221U) + 1376312589U;\n    return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);\n}\n\nuint hash( uint x ) {\n    x += ( x << 10u );\n    x ^= ( x >>  6u );\n    x += ( x <<  3u );\n    x ^= ( x >> 11u );\n    x += ( x << 15u );\n    return x;\n}\nfloat random( float f ) {\n    const uint mantissaMask = 0x007FFFFFu;\n    const uint one          = 0x3F800000u;\n\n    uint h = hash( floatBitsToUint( f ) );\n    h &= mantissaMask;\n    h |= one;\n\n    float  r2 = uintBitsToFloat( h );\n    return r2 - 1.0;\n}\n\n\n// #define HASHSCALE1 .1031\n// float random(float p) {\n// \tvec3 p3  = fract(vec3(p) * HASHSCALE1);\n//   p3 += dot(p3, p3.yzx + 19.19);\n//   return fract((p3.x + p3.y) * p3.z);\n// }\n\n// const vec2 randomConst = vec2(\n//   23.14069263277926, // e^pi (Gelfond's constant)\n//    2.665144142690225 // 2^sqrt(2) (Gelfond\u2013Schneider constant)\n// );\n// float random(float seed) {\n//     return fract(cos(dot(vec2(seed,seed), randomConst)) * 12345.6789);\n// }\n\n";
var distancesInitializationSource = "\n//Reads the distances and indices from the knn texture\nvoid initializeDistances(int pnt_id) {\n  //row coordinate in the texture\n  float row = (floor(float(pnt_id)/points_per_row_knn)+0.5)/num_rows_knn;\n  //column of the first neighbor\n  float start_col = mod(float(pnt_id),points_per_row_knn)*NUM_NEIGHBORS_FLOAT;\n  for(int n = 0; n < NUM_NEIGHBORS; n++) {\n    float col = (start_col+float(n)+0.5);\n    //normalized by the width of the texture\n    col /= (points_per_row_knn*NUM_NEIGHBORS_FLOAT);\n    //reads the index in the red channel and the distances in the green one\n    vec4 init = texture(starting_knn_tex,vec2(col,row));\n\n    knn_heap[n] = int(init.r);\n    distances_heap[n] = init.g;\n  }\n}\n";
var knnHeapSource = "\n//Swaps two points in the knn-heap\nvoid swap(int i, int j) {\n  float swap_value = distances_heap[i];\n  distances_heap[i] = distances_heap[j];\n  distances_heap[j] = swap_value;\n  int swap_id = knn_heap[i];\n  knn_heap[i] = knn_heap[j];\n  knn_heap[j] = swap_id;\n}\n\n//I can make use of the heap property but\n//I have to implement a recursive function\nbool inTheHeap(float dist_sq, int id) {\n  for(int i = 0; i < NUM_NEIGHBORS; ++i) {\n    if(knn_heap[i] == id) {\n      return true;\n    }\n  }\n  return false;\n}\n\nvoid insertInKNN(float dist_sq, int j) {\n  //not in the KNN\n  if(dist_sq >= distances_heap[0]) {\n    return;\n  }\n\n  //the point is already in the KNN\n  if(inTheHeap(dist_sq,j)) {\n    return;\n  }\n\n  //Insert in the new point in the root\n  distances_heap[0] = dist_sq;\n  knn_heap[0] = j;\n  //Sink procedure\n  int swap_id = 0;\n  while(swap_id*2+1 < NUM_NEIGHBORS) {\n    int left_id = swap_id*2+1;\n    int right_id = swap_id*2+2;\n    if(distances_heap[left_id] > distances_heap[swap_id] ||\n        (right_id < NUM_NEIGHBORS &&\n                            distances_heap[right_id] > distances_heap[swap_id])\n      ) {\n      if(distances_heap[left_id] > distances_heap[right_id]\n         || right_id >= NUM_NEIGHBORS) {\n        swap(swap_id,left_id);\n        swap_id = left_id;\n      }else{\n        swap(swap_id,right_id);\n        swap_id = right_id;\n      }\n    }else{\n      break;\n    }\n  }\n}\n";
var vertexPositionSource = "\n  //Line positions\n  float row = (floor(float(point_id)/points_per_row_knn)+0.5)/num_rows_knn;\n  row = row*2.0-1.0;\n  if(line_id < int(1)) {\n    //for the first vertex only the position is important\n    float col = (mod(float(point_id),points_per_row_knn))/(points_per_row_knn);\n    col = col*2.0-1.0;\n    gl_Position = vec4(col,row,0,1);\n    neighbor_id = 0.;\n    return;\n  }\n  //The computation of the KNN happens only for the second vertex\n  float col = (mod(float(point_id),points_per_row_knn)+1.)/(points_per_row_knn);\n  col = col*2.0-1.0;\n  gl_Position = vec4(col,row,0,1);\n";
function createBruteForceKNNProgram(gpgpu, numNeighbors, distanceComputationSource) {
    var vertexShaderSource = "#version 300 es\n    " +
        generateVariablesAndDeclarationsSource(numNeighbors) +
        distancesInitializationSource + distanceComputationSource +
        knnHeapSource + ("\n    void main() {\n      //Getting the id of the point and the line id (0/1)\n      point_id = int((vertex_id / 2.0) + 0.1);\n      int line_id = int(mod(vertex_id + 0.1, 2.));\n      if(float(point_id) >= num_points) {\n        return;\n      }\n\n      " + vertexPositionSource + "\n\n      //////////////////////////////////\n      //KNN computation\n      initializeDistances(point_id);\n      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {\n        //TODO make it more readable\n\n        int j = int(mod(\n                    float(point_id + i) //point id + current offset\n                    + iteration * float(NEIGH_PER_ITER) //iteration offset\n                    + 1.25,// +1 for avoid checking the point itself,\n                           // +0.25 for error compensation\n                    num_points\n                  ));\n        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j,j+1,j+2,j+3);\n        insertInKNN(dist_squared.r, j);\n        insertInKNN(dist_squared.g, j+1);\n        insertInKNN(dist_squared.b, j+2);\n        insertInKNN(dist_squared.a, j+3);\n      }\n\n      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {\n        knn[n].r = float(knn_heap[n*4]);\n        knn[n].g = float(knn_heap[n*4+1]);\n        knn[n].b = float(knn_heap[n*4+2]);\n        knn[n].a = float(knn_heap[n*4+3]);\n      }\n\n      neighbor_id = NUM_NEIGHBORS_FLOAT;\n    }\n  ");
    var knnFragmentShaderSource = generateFragmentShaderSource(distanceComputationSource, numNeighbors);
    return gl_util.createVertexProgram(gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
}
exports.createBruteForceKNNProgram = createBruteForceKNNProgram;
function createRandomSamplingKNNProgram(gpgpu, numNeighbors, distanceComputationSource) {
    var vertexShaderSource = "#version 300 es\n    " +
        generateVariablesAndDeclarationsSource(numNeighbors) +
        distancesInitializationSource + randomGeneratorSource +
        distanceComputationSource + knnHeapSource + ("\n    void main() {\n      //Getting the id of the point and the line id (0/1)\n      point_id = int((vertex_id/2.0)+0.1);\n      int line_id = int(mod(vertex_id+0.1,2.));\n      if(float(point_id) >= num_points) {\n        return;\n      }\n\n      " + vertexPositionSource + "\n\n      //////////////////////////////////\n      //KNN computation\n\n      initializeDistances(point_id);\n      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {\n        //BAD SEED\n        //uint seed\n        //= uint(float(point_id) + float(NEIGH_PER_ITER)*iteration + float(i));\n        //GOOD SEED\n        //uint seed\n        //= uint(float(point_id) + float(num_points)*iteration + float(i));\n\n        float seed\n            = float(float(point_id) + float(num_points)*iteration + float(i));\n        int j0 = int(random(seed)*num_points);\n        int j1 = int(random(seed+1.)*num_points);\n        int j2 = int(random(seed+2.)*num_points);\n        int j3 = int(random(seed+3.)*num_points);\n\n        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j0,j1,j2,j3);\n        if(j0!=point_id)insertInKNN(dist_squared.r, j0);\n        if(j1!=point_id)insertInKNN(dist_squared.g, j1);\n        if(j2!=point_id)insertInKNN(dist_squared.b, j2);\n        if(j3!=point_id)insertInKNN(dist_squared.a, j3);\n      }\n\n      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {\n        knn[n].r = float(knn_heap[n*4]);\n        knn[n].g = float(knn_heap[n*4+1]);\n        knn[n].b = float(knn_heap[n*4+2]);\n        knn[n].a = float(knn_heap[n*4+3]);\n      }\n      neighbor_id = NUM_NEIGHBORS_FLOAT;\n    }\n  ");
    var knnFragmentShaderSource = generateFragmentShaderSource(distanceComputationSource, numNeighbors);
    return gl_util.createVertexProgram(gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
}
exports.createRandomSamplingKNNProgram = createRandomSamplingKNNProgram;
function createKNNDescentProgram(gpgpu, numNeighbors, distanceComputationSource) {
    var vertexShaderSource = "#version 300 es\n    " +
        generateVariablesAndDeclarationsSource(numNeighbors) +
        distancesInitializationSource + randomGeneratorSource +
        distanceComputationSource + knnHeapSource + ("\n    int fetchNeighborIdFromKNNTexture(int id, int neighbor_id) {\n      //row coordinate in the texture\n      float row = (floor(float(id)/points_per_row_knn)+0.5)/num_rows_knn;\n      //column of the first neighbor\n      float start_col = mod(float(id),points_per_row_knn)*NUM_NEIGHBORS_FLOAT;\n      //column of the neighbor of interest\n      float col = (start_col+float(neighbor_id)+0.5);\n      //normalized by the width of the texture\n      col /= (points_per_row_knn*NUM_NEIGHBORS_FLOAT);\n      //reads the index in the red channel and the distances in the green one\n      vec4 knn_link = texture(starting_knn_tex,vec2(col,row));\n      //return the index\n      return int(knn_link.r);\n    }\n\n    int neighborOfANeighbor(int my_id, uint seed) {\n      //float random0 = hash1(seed);\n      float random0 = random(float(seed));\n      // random0 = random0*random0;\n      // random0 = 1. - random0;\n\n      //float random1 = hash1(seed*1798191U);\n      float random1 = random(float(seed+7U));\n      // random1 = random1*random1;\n      // random1 = 1. - random1;\n\n      //fetch a neighbor from the heap\n      int neighbor = knn_heap[int(random0*NUM_NEIGHBORS_FLOAT)];\n      //if it is not a valid pick a random point\n      if(neighbor < 0) {\n        return int(random(float(seed))*num_points);\n      }\n\n      //if it is valid I fetch from the knn graph texture one of its neighbors\n      int neighbor2ndDegree = fetchNeighborIdFromKNNTexture(\n                                    neighbor,int(random1*NUM_NEIGHBORS_FLOAT));\n      //if it is not a valid pick a random point\n      if(neighbor2ndDegree < 0) {\n        return int(random(float(seed))*num_points);\n      }\n      return neighbor2ndDegree;\n    }\n\n    void main() {\n      //Getting the id of the point and the line id (0/1)\n      point_id = int((vertex_id/2.0)+0.1);\n      int line_id = int(mod(vertex_id+0.1,2.));\n      if(float(point_id) >= num_points) {\n        return;\n      }\n      " + vertexPositionSource + "\n\n      //////////////////////////////////\n      //KNN computation\n      initializeDistances(point_id);\n      for(int i = 0; i < NEIGH_PER_ITER; i += 4) {\n        //BAD SEED\n        //uint seed\n        //= uint(float(point_id) + float(NEIGH_PER_ITER)*iteration + float(i));\n        //GOOD SEED\n        uint seed\n              = uint(float(point_id) + float(num_points)*iteration + float(i));\n        int j0 = neighborOfANeighbor(point_id,seed);\n        int j1 = neighborOfANeighbor(point_id,seed+1U);\n        int j2 = neighborOfANeighbor(point_id,seed+2U);\n        int j3 = neighborOfANeighbor(point_id,seed+3U);\n\n        vec4 dist_squared = pointDistanceSquaredBatch(point_id,j0,j1,j2,j3);\n        if(j0!=point_id)insertInKNN(dist_squared.r, j0);\n        if(j1!=point_id)insertInKNN(dist_squared.g, j1);\n        if(j2!=point_id)insertInKNN(dist_squared.b, j2);\n        if(j3!=point_id)insertInKNN(dist_squared.a, j3);\n      }\n\n      for(int n = 0; n < NUM_PACKED_NEIGHBORS; n++) {\n        knn[n].r = float(knn_heap[n*4]);\n        knn[n].g = float(knn_heap[n*4+1]);\n        knn[n].b = float(knn_heap[n*4+2]);\n        knn[n].a = float(knn_heap[n*4+3]);\n      }\n      neighbor_id = NUM_NEIGHBORS_FLOAT;\n    }\n  ");
    var knnFragmentShaderSource = generateFragmentShaderSource(distanceComputationSource, numNeighbors);
    return gl_util.createVertexProgram(gpgpu.gl, vertexShaderSource, knnFragmentShaderSource);
}
exports.createKNNDescentProgram = createKNNDescentProgram;
function executeKNNProgram(gpgpu, program, dataTex, startingKNNTex, iteration, knnShape, vertexIdBuffer, targetTex) {
    var gl = gpgpu.gl;
    var oldProgram = gpgpu.program;
    var oldLineWidth = gl.getParameter(gl.LINE_WIDTH);
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, knnShape.numRows, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    if (oldLineWidth !== 1) {
        gl.lineWidth(1);
    }
    gpgpu.setProgram(program);
    gl.clearColor(0., 0., 0., 0.);
    gl.clear(gl.COLOR_BUFFER_BIT);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexIdBuffer); });
    tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'vertex_id', vertexIdBuffer, 1, 0, 0);
    var dataTexLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'data_tex');
    gpgpu.setInputMatrixTexture(dataTex, dataTexLoc, 0);
    var startingKNNTexLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'starting_knn_tex');
    gpgpu.setInputMatrixTexture(startingKNNTex, startingKNNTexLoc, 1);
    var iterationLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'iteration');
    gl.uniform1f(iterationLoc, iteration);
    var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
    gl.uniform1f(numPointsLoc, knnShape.numPoints);
    var pntsPerRowKNNLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row_knn');
    gl.uniform1f(pntsPerRowKNNLoc, knnShape.pointsPerRow);
    var numRowsKNNLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows_knn');
    gl.uniform1f(numRowsKNNLoc, knnShape.numRows);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.drawArrays(gl.LINES, 0, knnShape.numPoints * 2); });
    if (oldProgram != null) {
        gpgpu.setProgram(oldProgram);
        tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
    }
    if (oldLineWidth !== 1) {
        gl.lineWidth(oldLineWidth);
    }
}
exports.executeKNNProgram = executeKNNProgram;
function createCopyDistancesProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D knn_tex;\n    uniform float width;\n    uniform float height;\n\n    void main() {\n      vec2 coordinates = gl_FragCoord.xy / vec2(width,height);\n      float distance = texture2D(knn_tex,coordinates).g;\n      gl_FragColor = vec4(distance,0,0,1);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createCopyDistancesProgram = createCopyDistancesProgram;
function executeCopyDistancesProgram(gpgpu, program, knnTex, knnShape, targetTex) {
    var gl = gpgpu.gl;
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, knnShape.numRows, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    gpgpu.setProgram(program);
    var knnLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_tex');
    gpgpu.setInputMatrixTexture(knnTex, knnLoc, 0);
    var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'width');
    gl.uniform1f(pntsPerRowLoc, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
    var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'height');
    gl.uniform1f(numRowsLoc, knnShape.numRows);
    gpgpu.executeProgram();
}
exports.executeCopyDistancesProgram = executeCopyDistancesProgram;
function createCopyIndicesProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D knn_tex;\n    uniform float width;\n    uniform float height;\n\n    void main() {\n      vec2 coordinates = gl_FragCoord.xy / vec2(width,height);\n      float id = texture2D(knn_tex,coordinates).r;\n      gl_FragColor = vec4(id,0,0,1);\n\n      if(id < 0.) {\n        gl_FragColor.b = 1.;\n      }\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createCopyIndicesProgram = createCopyIndicesProgram;
function executeCopyIndicesProgram(gpgpu, program, knnTex, knnShape, targetTex) {
    var gl = gpgpu.gl;
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, knnShape.numRows, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    gpgpu.setProgram(program);
    var knnLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_tex');
    gpgpu.setInputMatrixTexture(knnTex, knnLoc, 0);
    var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'width');
    gl.uniform1f(pntsPerRowLoc, knnShape.pointsPerRow * knnShape.pixelsPerPoint);
    var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'height');
    gl.uniform1f(numRowsLoc, knnShape.numRows);
    gpgpu.executeProgram();
}
exports.executeCopyIndicesProgram = executeCopyIndicesProgram;
//# sourceMappingURL=knn_util.js.map