"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var dataset_util = require("./dataset_util");
var gl_util = require("./gl_util");
var knn_util = require("./knn_util");
function instanceOfRearrangedData(object) {
    return 'numPoints' in object && 'pointsPerRow' in object &&
        'pixelsPerPoint' in object && 'numRows' in object;
}
function instanceOfCustomDataDefinition(object) {
    return 'distanceComputationCode' in object;
}
var KNNEstimator = (function () {
    function KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, numNeighs, verbose) {
        if (verbose != null) {
            this.verbose = verbose;
        }
        else {
            verbose = false;
        }
        this.backend = tf.ENV.findBackend('webgl');
        this.gpgpu = this.backend.getGPGPUContext();
        this._iteration = 0;
        this.dataTexture = dataTexture;
        if (numNeighs > 128) {
            throw new Error('kNN size must not be greater than 128');
        }
        if (numNeighs % 4 !== 0) {
            throw new Error('kNN size must be a multiple of 4');
        }
        this.numNeighs = numNeighs;
        var knnPointsPerRow = Math.ceil(Math.sqrt(numNeighs * numPoints) / numNeighs);
        this.knnDataShape = {
            numPoints: numPoints,
            pixelsPerPoint: numNeighs,
            pointsPerRow: knnPointsPerRow,
            numRows: Math.ceil(numPoints / knnPointsPerRow)
        };
        this.log('knn-pntsPerRow', this.knnDataShape.pointsPerRow);
        this.log('knn-numRows', this.knnDataShape.numRows);
        this.log('knn-pixelsPerPoint', this.knnDataShape.pixelsPerPoint);
        var distanceComputationSource;
        if (instanceOfRearrangedData(dataFormat)) {
            var rearrangedData = dataFormat;
            distanceComputationSource =
                dataset_util.generateDistanceComputationSource(rearrangedData);
        }
        else if (instanceOfCustomDataDefinition(dataFormat)) {
            var customDataDefinition = dataFormat;
            distanceComputationSource = customDataDefinition.distanceComputationCode;
        }
        this.initializeTextures();
        this.initilizeCustomWebGLPrograms(distanceComputationSource);
    }
    Object.defineProperty(KNNEstimator.prototype, "knnShape", {
        get: function () { return this.knnDataShape; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(KNNEstimator.prototype, "iteration", {
        get: function () { return this._iteration; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(KNNEstimator.prototype, "pointsPerIteration", {
        get: function () { return 20; },
        enumerable: true,
        configurable: true
    });
    KNNEstimator.prototype.log = function (str, obj) {
        if (this.verbose) {
            if (obj != null) {
                console.log(str + ": \t" + obj);
            }
            else {
                console.log(str);
            }
        }
    };
    KNNEstimator.prototype.initializeTextures = function () {
        var initNeigh = new Float32Array(this.knnDataShape.pointsPerRow *
            this.knnDataShape.pixelsPerPoint * 2 *
            this.knnDataShape.numRows);
        var numNeighs = this.knnDataShape.pixelsPerPoint;
        for (var i = 0; i < this.knnDataShape.numPoints; ++i) {
            for (var n = 0; n < numNeighs; ++n) {
                initNeigh[(i * numNeighs + n) * 2] = -1;
                initNeigh[(i * numNeighs + n) * 2 + 1] = 10e30;
            }
        }
        this.log('knn-textureWidth', this.knnDataShape.pointsPerRow *
            this.knnDataShape.pixelsPerPoint);
        this.log('knn-textureHeight', this.knnDataShape.numRows);
        this.knnTexture0 = gl_util.createAndConfigureTexture(this.gpgpu.gl, this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint, this.knnDataShape.numRows, 2, initNeigh);
        this.knnTexture1 = gl_util.createAndConfigureTexture(this.gpgpu.gl, this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint, this.knnDataShape.numRows, 2, initNeigh);
    };
    KNNEstimator.prototype.initilizeCustomWebGLPrograms = function (distanceComputationSource) {
        this.log('knn Create programs/buffers start');
        this.copyDistancesProgram = knn_util.createCopyDistancesProgram(this.gpgpu);
        this.log('knn Create indices program');
        this.copyIndicesProgram = knn_util.createCopyIndicesProgram(this.gpgpu);
        this.log('knn Create brute force knn program, numNeighs', this.numNeighs);
        this.bruteForceKNNProgram = knn_util.createBruteForceKNNProgram(this.gpgpu, this.numNeighs, distanceComputationSource);
        this.log('knn Create random sampling force knn program');
        this.randomSamplingKNNProgram = knn_util.createRandomSamplingKNNProgram(this.gpgpu, this.numNeighs, distanceComputationSource);
        this.log('knn Create descent program');
        this.kNNDescentProgram = knn_util.createKNNDescentProgram(this.gpgpu, this.numNeighs, distanceComputationSource);
        var linesVertexId = new Float32Array(this.knnDataShape.numPoints * 2);
        {
            for (var i = 0; i < this.knnDataShape.numPoints * 2; ++i) {
                linesVertexId[i] = i;
            }
        }
        this.log('knn Create static vertex start');
        this.linesVertexIdBuffer = tf.webgl.webgl_util.createStaticVertexBuffer(this.gpgpu.gl, linesVertexId);
        this.log('knn Create programs/buffers done');
    };
    KNNEstimator.prototype.iterateBruteForce = function () {
        if ((this._iteration % 2) === 0) {
            this.iterateGPU(this.dataTexture, this._iteration, this.knnTexture0, this.knnTexture1);
        }
        else {
            this.iterateGPU(this.dataTexture, this._iteration, this.knnTexture1, this.knnTexture0);
        }
        ++this._iteration;
        this.gpgpu.gl.finish();
    };
    KNNEstimator.prototype.iterateRandomSampling = function () {
        if ((this._iteration % 2) === 0) {
            this.iterateRandomSamplingGPU(this.dataTexture, this._iteration, this.knnTexture0, this.knnTexture1);
        }
        else {
            this.iterateRandomSamplingGPU(this.dataTexture, this._iteration, this.knnTexture1, this.knnTexture0);
        }
        ++this._iteration;
        this.gpgpu.gl.finish();
    };
    KNNEstimator.prototype.iterateKNNDescent = function () {
        if ((this._iteration % 2) === 0) {
            this.iterateKNNDescentGPU(this.dataTexture, this._iteration, this.knnTexture0, this.knnTexture1);
        }
        else {
            this.iterateKNNDescentGPU(this.dataTexture, this._iteration, this.knnTexture1, this.knnTexture0);
        }
        ++this._iteration;
        this.gpgpu.gl.finish();
    };
    KNNEstimator.prototype.knn = function () {
        if ((this._iteration % 2) === 0) {
            return this.knnTexture0;
        }
        else {
            return this.knnTexture1;
        }
    };
    KNNEstimator.prototype.distancesTensor = function () {
        var _this = this;
        return tf.tidy(function () {
            var distances = tf.zeros([
                _this.knnDataShape.numRows,
                _this.knnDataShape.pointsPerRow * _this.knnDataShape.pixelsPerPoint
            ]);
            var knnTexture = _this.knn();
            knn_util.executeCopyDistancesProgram(_this.gpgpu, _this.copyDistancesProgram, knnTexture, _this.knnDataShape, _this.backend.getTexture(distances.dataId));
            return distances
                .reshape([
                _this.knnDataShape.numRows * _this.knnDataShape.pointsPerRow,
                _this.knnDataShape.pixelsPerPoint
            ])
                .slice([0, 0], [
                _this.knnDataShape.numPoints, _this.knnDataShape.pixelsPerPoint
            ]);
        });
    };
    KNNEstimator.prototype.indicesTensor = function () {
        var _this = this;
        return tf.tidy(function () {
            var indices = tf.zeros([
                _this.knnDataShape.numRows,
                _this.knnDataShape.pointsPerRow * _this.knnDataShape.pixelsPerPoint
            ]);
            var knnTexture = _this.knn();
            knn_util.executeCopyIndicesProgram(_this.gpgpu, _this.copyIndicesProgram, knnTexture, _this.knnDataShape, _this.backend.getTexture(indices.dataId));
            return indices
                .reshape([
                _this.knnDataShape.numRows * _this.knnDataShape.pointsPerRow,
                _this.knnDataShape.pixelsPerPoint
            ])
                .slice([0, 0], [
                _this.knnDataShape.numPoints, _this.knnDataShape.pixelsPerPoint
            ]);
        });
    };
    KNNEstimator.prototype.forceFlush = function () {
        var mat0 = this.downloadTextureToMatrix(this.knnTexture0);
        console.log("Flush: " + mat0.length / mat0.length);
    };
    KNNEstimator.prototype.downloadTextureToMatrix = function (texture) {
        return this.gpgpu.downloadMatrixFromTexture(texture, this.knnDataShape.numRows, this.knnDataShape.pointsPerRow * this.knnDataShape.pixelsPerPoint);
    };
    KNNEstimator.prototype.iterateGPU = function (dataTexture, _iteration, startingKNNTexture, targetTexture) {
        knn_util.executeKNNProgram(this.gpgpu, this.bruteForceKNNProgram, dataTexture, startingKNNTexture, _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
    };
    KNNEstimator.prototype.iterateRandomSamplingGPU = function (dataTexture, _iteration, startingKNNTexture, targetTexture) {
        knn_util.executeKNNProgram(this.gpgpu, this.randomSamplingKNNProgram, dataTexture, startingKNNTexture, _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
    };
    KNNEstimator.prototype.iterateKNNDescentGPU = function (dataTexture, _iteration, startingKNNTexture, targetTexture) {
        knn_util.executeKNNProgram(this.gpgpu, this.kNNDescentProgram, dataTexture, startingKNNTexture, _iteration, this.knnDataShape, this.linesVertexIdBuffer, targetTexture);
    };
    return KNNEstimator;
}());
exports.KNNEstimator = KNNEstimator;
//# sourceMappingURL=knn.js.map