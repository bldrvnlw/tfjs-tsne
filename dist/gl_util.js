"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
function createVertexProgram(gl, vertexShaderSource, fragmentShaderSource) {
    var vertexShader = tf.webgl.webgl_util.createVertexShader(gl, vertexShaderSource);
    var fragmentShader = tf.webgl.webgl_util.createFragmentShader(gl, fragmentShaderSource);
    var program = tf.webgl.webgl_util.createProgram(gl);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, vertexShader); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, fragmentShader); });
    tf.webgl.webgl_util.linkProgram(gl, program);
    tf.webgl.webgl_util.validateProgram(gl, program);
    return program;
}
exports.createVertexProgram = createVertexProgram;
function createAndConfigureInterpolatedTexture(gl, width, height, numChannels, pixels) {
    tf.webgl.webgl_util.validateTextureSize(gl, width, height);
    var texture = tf.webgl.webgl_util.createTexture(gl);
    var tex2d = gl.TEXTURE_2D;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    var format = getTextureFormat(gl, numChannels);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.LINEAR); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.LINEAR); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, getTextureType(gl), pixels); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    return texture;
}
exports.createAndConfigureInterpolatedTexture = createAndConfigureInterpolatedTexture;
function createAndConfigureTexture(gl, width, height, numChannels, pixels) {
    tf.webgl.webgl_util.validateTextureSize(gl, width, height);
    var texture = tf.webgl.webgl_util.createTexture(gl);
    var tex2d = gl.TEXTURE_2D;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    var format = getTextureFormat(gl, numChannels);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, getTextureType(gl), pixels); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    return texture;
}
exports.createAndConfigureTexture = createAndConfigureTexture;
function createAndConfigureUByteTexture(gl, width, height, numChannels, pixels) {
    tf.webgl.webgl_util.validateTextureSize(gl, width, height);
    var texture = tf.webgl.webgl_util.createTexture(gl);
    var tex2d = gl.TEXTURE_2D;
    var internalFormat = getTextureInternalUByteFormat(gl, numChannels);
    var format = getTextureFormat(gl, numChannels);
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, getTextureTypeUByte(gl), pixels); });
    tf.webgl.webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    return texture;
}
exports.createAndConfigureUByteTexture = createAndConfigureUByteTexture;
function getTextureInternalFormat(gl, numChannels) {
    if (numChannels === 4) {
        return gl.RGBA32F;
    }
    else if (numChannels === 3) {
        return gl.RGB32F;
    }
    else if (numChannels === 2) {
        return gl.RG32F;
    }
    return gl.R32F;
}
function getTextureInternalUByteFormat(gl, numChannels) {
    if (numChannels === 4) {
        return gl.RGBA8;
    }
    else if (numChannels === 3) {
        return gl.RGB8;
    }
    else if (numChannels === 2) {
        return gl.RG8;
    }
    return gl.R8;
}
function getTextureFormat(gl, numChannels) {
    if (numChannels === 4) {
        return gl.RGBA;
    }
    else if (numChannels === 3) {
        return gl.RGB;
    }
    else if (numChannels === 2) {
        return gl.RG;
    }
    return gl.RED;
}
function getTextureType(gl) {
    return gl.FLOAT;
}
function getTextureTypeUByte(gl) {
    return gl.UNSIGNED_BYTE;
}
//# sourceMappingURL=gl_util.js.map