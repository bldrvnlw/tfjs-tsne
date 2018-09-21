"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var gl_util = require("./gl_util");
function createEmbeddingSplatterProgram(gpgpu) {
    var vertexShaderSource = "#version 300 es\n    precision highp float;\n    in float vertex_id;\n\n    uniform sampler2D embedding_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float kernel_support;\n    uniform float points_per_row;\n    uniform float num_rows;\n\n    out vec2 kernel_coords;\n\n    void main() {\n      //TODO Clean up and check performance loss due to the conversions\n      uint pnt_id = uint((vertex_id / 4.0) + 0.1);\n      uint quad_id = uint(mod(vertex_id + 0.1,4.));\n\n      uint row    = uint((float(pnt_id) + 0.1)/points_per_row);\n      uint column = uint(float(pnt_id) - float(row) * points_per_row);\n\n      float width = (points_per_row * 2.0);\n      float row_tex = (float(row) + 0.5) / num_rows;\n      vec2 tex_coords_x = vec2((float(column) * 2. + 0.5) / width, row_tex);\n      vec2 tex_coords_y = vec2((float(column) * 2. + 1.5) / width, row_tex);\n\n      float x_pnt = texture(embedding_tex,tex_coords_x).r;\n      float y_pnt = texture(embedding_tex,tex_coords_y).r;\n      vec2 vertex_coords = vec2(x_pnt,y_pnt);\n\n      if(quad_id == uint(0)) {kernel_coords = vec2(-1,-1);}\n      else if(quad_id == uint(1)) {kernel_coords = vec2(1,-1);}\n      else if(quad_id == uint(2)) {kernel_coords = vec2(1,1);}\n      else if(quad_id == uint(3)) {kernel_coords = vec2(-1,1);}\n\n      vertex_coords += kernel_coords * kernel_support;      // embedding space\n      vertex_coords = (vertex_coords - minV) / (maxV-minV); //  0:1 space\n      vertex_coords = vertex_coords * 2.0 - 1.0;            // -1:1 space\n\n      gl_Position = vec4(vertex_coords,0,1);\n    }\n  ";
    var fragmentShaderSource = "#version 300 es\n    precision highp float;\n    uniform sampler2D kernel_tex;\n    in vec2 kernel_coords;\n    out vec4 fragmentColor;\n\n    void main() {\n      fragmentColor = texture(kernel_tex,(kernel_coords + 1.) / 2.0);\n    }\n  ";
    return gl_util.createVertexProgram(gpgpu.gl, vertexShaderSource, fragmentShaderSource);
}
exports.createEmbeddingSplatterProgram = createEmbeddingSplatterProgram;
function executeEmbeddingSplatterProgram(gpgpu, program, targetTex, embeddingTex, kernelTex, targetTexDiameter, numPoints, minX, minY, maxX, maxY, kernelSupport, pntsPerRow, numRows, vertexIdBuffer) {
    var gl = gpgpu.gl;
    var oldProgram = gpgpu.program;
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, targetTexDiameter, targetTexDiameter);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    gpgpu.setProgram(program);
    gl.clearColor(0., 0., 0., 0.);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexIdBuffer); });
    tf.webgl.webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'vertex_id', vertexIdBuffer, 1, 0, 0);
    var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
    gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
    var kernelLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'kernel_tex');
    gpgpu.setInputMatrixTexture(kernelTex, kernelLocation, 1);
    var kernelSupportLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'kernel_support');
    gl.uniform1f(kernelSupportLoc, kernelSupport);
    var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
    gl.uniform1f(pntsPerRowLoc, pntsPerRow);
    var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
    gl.uniform1f(numRowsLoc, numRows);
    var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
    gl.uniform2f(minLoc, minX, minY);
    var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
    gl.uniform2f(maxLoc, maxX, maxY);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.drawArrays(gl.TRIANGLES, 0, numPoints * 2 * 3); });
    gl.disable(gl.BLEND);
    if (oldProgram != null) {
        gpgpu.setProgram(oldProgram);
        tf.webgl.gpgpu_util.bindVertexProgramAttributeStreams(gpgpu.gl, oldProgram, gpgpu.vertexBuffer);
    }
}
exports.executeEmbeddingSplatterProgram = executeEmbeddingSplatterProgram;
function createQInterpolatorProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D embedding_tex;\n    uniform sampler2D splat_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n\n    void main() {\n      vec2 pnt_location = gl_FragCoord.xy - vec2(0.5,0.5);\n\n      if(pnt_location.y * points_per_row + pnt_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n\n      float emb_width = (points_per_row * 2.0);\n      float emb_row_coord = (pnt_location.y + 0.5) / num_rows;\n      vec2 emb_coords_x\n              = vec2((pnt_location.x * 2.+0.5) / emb_width, emb_row_coord);\n      vec2 emb_coords_y\n              = vec2((pnt_location.x * 2. + 1.5) / emb_width, emb_row_coord);\n\n      float x_pnt = texture2D(embedding_tex,emb_coords_x).r;\n      float y_pnt = texture2D(embedding_tex,emb_coords_y).r;\n\n      vec2 splat_coords = vec2(x_pnt,y_pnt);\n      splat_coords = (splat_coords - minV) / (maxV - minV); //  0:1 space\n\n      float q = (texture2D(splat_tex,splat_coords).r - 1.);\n\n      gl_FragColor = vec4(q, 0, 0, 1);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createQInterpolatorProgram = createQInterpolatorProgram;
function executeQInterpolatorProgram(gpgpu, program, splatTex, embeddingTex, numPoints, minX, minY, maxX, maxY, pntsPerRow, numRows, targetTex) {
    var gl = gpgpu.gl;
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    gpgpu.setProgram(program);
    var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
    gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
    var splatLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'splat_tex');
    gpgpu.setInputMatrixTexture(splatTex, splatLocation, 1);
    var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
    gl.uniform1f(pntsPerRowLoc, pntsPerRow);
    var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
    gl.uniform1f(numRowsLoc, numRows);
    var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
    gl.uniform1f(numPointsLoc, numPoints);
    var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
    gl.uniform2f(minLoc, minX, minY);
    var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
    gl.uniform2f(maxLoc, maxX, maxY);
    gpgpu.executeProgram();
}
exports.executeQInterpolatorProgram = executeQInterpolatorProgram;
function createXYInterpolatorProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D embedding_tex;\n    uniform sampler2D splat_tex;\n    uniform vec2 minV;\n    uniform vec2 maxV;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float eta;\n\n    void main() {\n      vec2 pnt_location = gl_FragCoord.xy - vec2(0.5,0.5);\n      pnt_location.x = floor(pnt_location.x/2.+0.1);\n\n      if(pnt_location.y*points_per_row + pnt_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n\n      float emb_width = (points_per_row * 2.0);\n      float emb_row_coord = (pnt_location.y + 0.5) / num_rows;\n      vec2 emb_coords_x\n              = vec2((pnt_location.x * 2. + 0.5) / emb_width, emb_row_coord);\n      vec2 emb_coords_y\n              = vec2((pnt_location.x * 2. + 1.5) / emb_width, emb_row_coord);\n\n      float x_pnt = texture2D(embedding_tex,emb_coords_x).r;\n      float y_pnt = texture2D(embedding_tex,emb_coords_y).r;\n\n      vec2 splat_coords = vec2(x_pnt,y_pnt);\n      splat_coords = (splat_coords - minV) / (maxV - minV); //  0:1 space\n\n      float q = 0.;\n      if(mod(gl_FragCoord.x - 0.5,2.) < 0.5 ) {\n        q = texture2D(splat_tex,splat_coords).g * eta * 2.;\n      }else{\n        q = texture2D(splat_tex,splat_coords).b * eta * 2.;\n      }\n\n      gl_FragColor = vec4(q,0.0,0.0,1);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createXYInterpolatorProgram = createXYInterpolatorProgram;
function executeXYInterpolatorProgram(gpgpu, program, splatTex, embeddingTex, targetTex, numPoints, minX, minY, maxX, maxY, pntsPerRow, numRows, eta) {
    var gl = gpgpu.gl;
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    gpgpu.setProgram(program);
    var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
    gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 0);
    var splatLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'splat_tex');
    gpgpu.setInputMatrixTexture(splatTex, splatLocation, 1);
    var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
    gl.uniform1f(pntsPerRowLoc, pntsPerRow);
    var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
    gl.uniform1f(numRowsLoc, numRows);
    var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
    gl.uniform1f(numPointsLoc, numPoints);
    var etaLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'eta');
    gl.uniform1f(etaLoc, eta);
    var minLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'minV');
    gl.uniform2f(minLoc, minX, minY);
    var maxLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'maxV');
    gl.uniform2f(maxLoc, maxX, maxY);
    gpgpu.executeProgram();
}
exports.executeXYInterpolatorProgram = executeXYInterpolatorProgram;
function createAttractiveForcesComputationProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n\n    uniform sampler2D embedding_tex;\n    uniform sampler2D offset_tex;\n    uniform sampler2D neigh_id_tex;\n    uniform sampler2D neigh_prob_tex;\n\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float num_neighs_per_row;\n    uniform float eta;\n\n    void main() {\n      //add for nearest pixel interpolation\n      vec2 half_pxl = vec2(0.5,0.5);\n\n      // Dimension of the fragment\n      // 0 -> x :1 -> y\n      float dimension = mod(gl_FragCoord.x - 0.4,2.);\n\n      //Point location in the [points_per_row,num_rows] space\n      vec2 i_location = gl_FragCoord.xy - half_pxl;\n      i_location.x = floor(i_location.x / 2. + 0.1);\n\n      //just an extra fragment -> return\n      if(i_location.y*points_per_row + i_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n\n      //Offset coordinates for the point\n      vec2 offset_coord = (i_location + half_pxl) /\n                                              vec2(points_per_row,num_rows);\n      //Offset information ...\n      vec4 offset_info  = texture2D(offset_tex,offset_coord);\n      //... contains the number of neighbors for the point ...\n      float num_neighs  = offset_info.z;\n      //... and the coordinates of the firts neigh in the neigh textures\n      vec2 offset_neigh = offset_info.xy;\n\n      //Computing the coordinates of the point in the texture\n      //_i represent the point to move, _j the neighbors\n      float emb_width = (points_per_row * 2.0);\n      float emb_row_i = (i_location.y + 0.5) / num_rows;\n      vec2 x_i_coord = vec2((i_location.x * 2. + 0.5) / emb_width, emb_row_i);\n      vec2 y_i_coord = vec2((i_location.x * 2. + 1.5) / emb_width, emb_row_i);\n      //getting the coordinates in the embedding\n      float x_i = texture2D(embedding_tex,x_i_coord).r;\n      float y_i = texture2D(embedding_tex,y_i_coord).r;\n\n      //Sum of all attractive forces\n      float sum_pos = 0.;\n\n      //Can't be higher than 1000 (perplexity is usually around 30)\n      //and a 'while' can't be used\n      for(int n = 0; n < 2000; ++n) {\n        //Actual check on number of neighbors\n        if(float(n) >= num_neighs) {\n          break;\n        }\n\n        //Get the id and the probability for the neighbor\n        float pij = texture2D(neigh_prob_tex,\n                              (offset_neigh + half_pxl) / num_neighs_per_row\n                             ).r;\n        float neigh_id = texture2D(neigh_id_tex,\n                                  (offset_neigh + half_pxl) / num_neighs_per_row\n                                  ).r;\n\n        //Getting the coordinates of the neighbor\n        vec2 j_location = vec2(mod(neigh_id + 0.1, points_per_row),\n                               floor(neigh_id / points_per_row + 0.1));\n        float emb_row_j = (j_location.y + 0.5) / num_rows;\n        vec2 x_j_coord = vec2((j_location.x * 2. + 0.5) / emb_width, emb_row_j);\n        vec2 y_j_coord = vec2((j_location.x * 2. + 1.5) / emb_width, emb_row_j);\n        float x_j = texture2D(embedding_tex,x_j_coord).r;\n        float y_j = texture2D(embedding_tex,y_j_coord).r;\n\n        //Actual computation of the attractive forces\n        float dist_x    = (x_i - x_j);\n        float dist_y    = (y_i - y_j);\n        float qij       = 1. / (1. + dist_x * dist_x + dist_y * dist_y);\n        //the update depends on the dimension that this fragment represents\n        if(dimension < 0.5) {\n          // * 4 / (num_points*2) -> * 2 / num_points\n          sum_pos += eta * 2. * pij * qij * dist_x / (num_points);\n        }else{\n          sum_pos += eta * 2. * pij * qij * dist_y / (num_points);\n        }\n\n        //Increase the coordinate of the neigh in the neigh_id texture\n        offset_neigh.x += 1.;\n        //check if the new neigh is in the next row\n        if(offset_neigh.x + 0.2 > num_neighs_per_row) {\n          //in that case reset the column and increase the row\n          offset_neigh.x = 0.1;\n          offset_neigh.y += 1.0;\n        }\n      }\n\n      //The output is the sum of the attractive forces\n      gl_FragColor = vec4(sum_pos,0,0,0);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createAttractiveForcesComputationProgram = createAttractiveForcesComputationProgram;
function executeAttractiveForcesComputationProgram(gpgpu, program, embeddingTex, offsetTex, neighIdTex, neighProbTex, numPoints, neighsPerRow, pntsPerRow, numRows, eta, targetTex) {
    var gl = gpgpu.gl;
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    gpgpu.setProgram(program);
    var embeddingLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'embedding_tex');
    gpgpu.setInputMatrixTexture(embeddingTex, embeddingLocation, 3);
    var offsetLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'offset_tex');
    gpgpu.setInputMatrixTexture(offsetTex, offsetLocation, 2);
    var neighIdLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'neigh_id_tex');
    gpgpu.setInputMatrixTexture(neighIdTex, neighIdLocation, 1);
    var neighProbLocation = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'neigh_prob_tex');
    gpgpu.setInputMatrixTexture(neighProbTex, neighProbLocation, 0);
    var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
    gl.uniform1f(numRowsLoc, numRows);
    var etaLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'eta');
    gl.uniform1f(etaLoc, eta);
    var neighsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_neighs_per_row');
    gl.uniform1f(neighsPerRowLoc, neighsPerRow);
    var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
    gl.uniform1f(numPointsLoc, numPoints);
    var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
    gl.uniform1f(pntsPerRowLoc, pntsPerRow);
    gpgpu.executeProgram();
}
exports.executeAttractiveForcesComputationProgram = executeAttractiveForcesComputationProgram;
function createEmbeddingInitializationProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n\n    uniform sampler2D random_tex;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n\n    void main() {\n      //add for nearest pixel interpolation\n      vec2 half_pxl = vec2(0.5,0.5);\n\n      // Dimension of the fragment\n      // 0 -> x :1 -> y\n      float dimension = mod(gl_FragCoord.x - 0.4,2.);\n      vec2 pnt_location = gl_FragCoord.xy - half_pxl;\n      pnt_location.x = floor(pnt_location.x / 2.);\n\n      //just an extra fragment -> return\n      if(pnt_location.y*points_per_row + pnt_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,1);\n        return;\n      }\n\n      float width = (points_per_row * 2.0);\n      float row_coord = (pnt_location.y + 0.5)/num_rows;\n      vec2 rad_coord = vec2((pnt_location.x * 2. + 0.5) / width, row_coord);\n      vec2 ang_coord = vec2((pnt_location.x * 2. + 1.5) / width, row_coord);\n\n      float rad = texture2D(random_tex,rad_coord).r * 3.;\n      float ang = texture2D(random_tex,ang_coord).r * 3.1415 * 2.;\n\n      gl_FragColor = vec4(rad,ang,0,1);\n\n      if(dimension < 0.5) {\n        gl_FragColor = vec4(cos(ang) * rad,0,0,0);\n      }else{\n        gl_FragColor = vec4(sin(ang) * rad,0,0,0);\n      }\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createEmbeddingInitializationProgram = createEmbeddingInitializationProgram;
function executeEmbeddingInitializationProgram(gpgpu, program, randomTex, numPoints, pntsPerRow, numRows, targetTex) {
    var gl = gpgpu.gl;
    if (targetTex != null) {
        gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * 2);
    }
    else {
        tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    }
    gpgpu.setProgram(program);
    var randomLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'random_tex');
    gpgpu.setInputMatrixTexture(randomTex, randomLoc, 3);
    var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
    gl.uniform1f(numRowsLoc, numRows);
    var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
    gl.uniform1f(numPointsLoc, numPoints);
    var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
    gl.uniform1f(pntsPerRowLoc, pntsPerRow);
    gpgpu.executeProgram();
}
exports.executeEmbeddingInitializationProgram = executeEmbeddingInitializationProgram;
function createDistributionParametersComputationProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n\n    #define MAX_NEIGHBORS 128\n    #define MAX_ITERATIONS 500\n    #define FLOAT_MAX 10e30\n    #define TOLERANCE 1e-5\n\n    uniform sampler2D knn_graph_tex;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float num_neighs;\n    uniform float perplexity;\n\n    vec2 half_pixel = vec2(0.5,0.5);\n    float distances_squared[MAX_NEIGHBORS];\n\n    void readDistances(vec2 point_location) {\n      for(int n = 0; n < MAX_NEIGHBORS; ++n ) {\n        if(float(n) >= num_neighs-0.1) {\n          break;\n        }\n        vec2 knn_coordinates = vec2(\n            (point_location.x * num_neighs + float(n) + half_pixel.x)\n                                        /(points_per_row * num_neighs),\n            (point_location.y + half_pixel.y) / num_rows\n        );\n        distances_squared[n] = texture2D(knn_graph_tex,knn_coordinates).g;\n      }\n    }\n\n    void main() {\n      vec2 point_location = gl_FragCoord.xy - half_pixel;\n      //invalid points\n      if(point_location.y*points_per_row + point_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n      readDistances(point_location);\n\n      //Beta computation\n      float beta = 1.;\n      float max_beta = FLOAT_MAX;\n      float min_beta = -FLOAT_MAX;\n      //To avoid computing the log at every iteration\n      float log_perplexity = log(perplexity);\n      float entropy_diff = 0.;\n      float entropy = 0.;\n      float sum_probabilities = 0.;\n\n      //Binary search for a maximum of MAX_ITERATIONS\n      for(int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {\n        //At every iteration I compute the\n        //entropy enforced by the current beta\n        sum_probabilities = 0.;\n        entropy = 0.;\n        for(int n = 0; n < MAX_NEIGHBORS; ++n ) {\n          if(float(n) >= num_neighs-0.1) {\n            break;\n          }\n          float neigh_probability = exp(-beta * distances_squared[n]);\n          sum_probabilities += neigh_probability;\n          entropy += beta * distances_squared[n] * neigh_probability;\n        }\n\n        entropy = entropy / sum_probabilities + log(sum_probabilities);\n        entropy_diff = entropy - log_perplexity;\n\n        //the current beta is good enough!\n        if(entropy_diff < TOLERANCE && -entropy_diff < TOLERANCE) {\n          break;\n        }\n\n        if(entropy_diff > 0.) {\n          min_beta = beta;\n          if(max_beta == FLOAT_MAX || max_beta == -FLOAT_MAX) {\n            beta *= 2.;\n          }else{\n            beta = (beta + max_beta) / 2.;\n          }\n        }else{\n          max_beta = beta;\n          if(min_beta == -FLOAT_MAX || min_beta == FLOAT_MAX) {\n            beta /= 2.;\n          }else{\n            beta = (beta + min_beta) / 2.;\n          }\n        }\n      }\n      gl_FragColor = vec4(beta,sum_probabilities,0,1);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createDistributionParametersComputationProgram = createDistributionParametersComputationProgram;
function executeDistributionParametersComputationProgram(gpgpu, program, knnGraph, numPoints, numNeighs, pntsPerRow, numRows, perplexity, targetTex) {
    var gl = gpgpu.gl;
    try {
        if (targetTex != null) {
            gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow);
        }
        else {
            tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
        }
        gpgpu.setProgram(program);
        var knnGraphLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_graph_tex');
        gpgpu.setInputMatrixTexture(knnGraph, knnGraphLoc, 0);
        var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
        gl.uniform1f(numRowsLoc, numRows);
        var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
        gl.uniform1f(numPointsLoc, numPoints);
        var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
        gl.uniform1f(pntsPerRowLoc, pntsPerRow);
        var numNeighsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_neighs');
        gl.uniform1f(numNeighsLoc, numNeighs);
        var perplexityLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'perplexity');
        gl.uniform1f(perplexityLoc, perplexity);
        gpgpu.executeProgram();
    }
    catch (e) {
        console.log('Error in executeDistributionParametersComputationProgram' +
            e.toString());
    }
}
exports.executeDistributionParametersComputationProgram = executeDistributionParametersComputationProgram;
function createGaussiaDistributionsFromDistancesProgram(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D knn_graph_tex;\n    uniform sampler2D parameters_tex;\n    uniform float points_per_row;\n    uniform float num_rows;\n    uniform float num_points;\n    uniform float num_neighs;\n\n    vec2 half_pixel = vec2(0.5,0.5);\n\n    void main() {\n      vec2 point_location = gl_FragCoord.xy - half_pixel;\n      point_location.x = floor(point_location.x / num_neighs);\n\n      //invalid points\n      if(point_location.y*points_per_row + point_location.x >= num_points) {\n        gl_FragColor = vec4(0,0,0,0);\n        return;\n      }\n      float distance_squared\n            = texture2D(knn_graph_tex,\n                        gl_FragCoord.xy /\n                        vec2(points_per_row*num_neighs,num_rows)\n                      ).g;\n      vec2 parameters\n            = texture2D(parameters_tex,\n                        (point_location.xy + half_pixel)/\n                        vec2(points_per_row,num_rows)\n                      ).rg;\n      float beta = parameters.r;\n      float normalization = parameters.g;\n\n      float probability = exp(-beta * distance_squared) / normalization;\n      //check for NaN for degenerated knn (d = 0 for every point)\n      if (!(probability < 0.0 || 0.0 < probability || probability == 0.0)) {\n        probability = 0.;\n      }\n\n      gl_FragColor = vec4(probability,0,0,1);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.createGaussiaDistributionsFromDistancesProgram = createGaussiaDistributionsFromDistancesProgram;
function executeGaussiaDistributionsFromDistancesProgram(gpgpu, program, knnGraph, parameters, numPoints, numNeighs, pntsPerRow, numRows, targetTex) {
    var gl = gpgpu.gl;
    try {
        gpgpu.enableAutomaticDebugValidation(true);
        if (targetTex != null) {
            gpgpu.setOutputMatrixTexture(targetTex, numRows, pntsPerRow * numNeighs);
        }
        else {
            tf.webgl.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
        }
        gpgpu.setProgram(program);
        var knnGraphLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'knn_graph_tex');
        gpgpu.setInputMatrixTexture(knnGraph, knnGraphLoc, 0);
        var parametersLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'parameters_tex');
        gpgpu.setInputMatrixTexture(parameters, parametersLoc, 1);
        var numRowsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_rows');
        gl.uniform1f(numRowsLoc, numRows);
        var numPointsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_points');
        gl.uniform1f(numPointsLoc, numPoints);
        var pntsPerRowLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'points_per_row');
        gl.uniform1f(pntsPerRowLoc, pntsPerRow);
        var numNeighsLoc = tf.webgl.webgl_util.getProgramUniformLocationOrThrow(gl, program, 'num_neighs');
        gl.uniform1f(numNeighsLoc, numNeighs);
        console.log('Execute Gaussion Dist from Distances');
        gpgpu.executeProgram();
    }
    catch (e) {
        console.log('Error executing Gaussian Dist From Distances Program ' +
            e.toString());
    }
}
exports.executeGaussiaDistributionsFromDistancesProgram = executeGaussiaDistributionsFromDistancesProgram;
//# sourceMappingURL=tsne_optimizer_util.js.map