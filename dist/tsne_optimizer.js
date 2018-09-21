"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var gl_util = require("./gl_util");
var knn_util = require("./knn_util");
var tsne_util = require("./tsne_optimizer_util");
var TSNEOptimizer = (function () {
    function TSNEOptimizer(numPoints, verbose, splatTextureDiameter, kernelTextureRadius) {
        if (verbose != null) {
            this.verbose = verbose;
        }
        else {
            verbose = false;
        }
        this.log('Initializing the tSNE gradient descent computation...');
        this.numPoints = numPoints;
        this._iteration = 0;
        var webglVersion = tf.ENV.get('WEBGL_VERSION');
        if (webglVersion === 1) {
            throw Error('WebGL version 1 is not supported by tfjs-tsne');
        }
        this.backend = tf.ENV.findBackend('webgl');
        if (this.backend === null) {
            throw Error('WebGL backend is not available');
        }
        this.gpgpu = this.backend.getGPGPUContext();
        tf.webgl.webgl_util.getExtensionOrThrow(this.gpgpu.gl, 'OES_texture_float_linear');
        this.pointsPerRow = Math.ceil(Math.sqrt(numPoints * 2));
        if (this.pointsPerRow % 2 === 1) {
            ++this.pointsPerRow;
        }
        this.pointsPerRow /= 2;
        this.numRows = Math.ceil(numPoints / this.pointsPerRow);
        this.log('\t# points per row', this.pointsPerRow);
        this.log('\t# rows', this.numRows);
        this._eta = 2500;
        this._momentum = tf.scalar(0.8);
        this.rawExaggeration =
            [{ iteration: 200, value: 4 }, { iteration: 600, value: 1 }];
        this.updateExaggeration();
        if (splatTextureDiameter == null) {
            splatTextureDiameter = 5;
        }
        this.splatTextureDiameter = splatTextureDiameter;
        if (kernelTextureRadius == null) {
            kernelTextureRadius = 50;
        }
        this.kernelTextureDiameter = kernelTextureRadius * 2 + 1;
        this.initializeRepulsiveForceTextures();
        this.log('\tSplat texture diameter', this.splatTextureDiameter);
        this.log('\tKernel texture diameter', this.kernelTextureDiameter);
        this.initilizeCustomWebGLPrograms();
        this.initializeEmbedding();
        this.log('\tEmbedding', this.embedding);
        this.log('\tGradient', this.gradient);
    }
    Object.defineProperty(TSNEOptimizer.prototype, "minX", {
        get: function () { return this._minX; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "maxX", {
        get: function () { return this._maxX; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "minY", {
        get: function () { return this._minY; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "maxY", {
        get: function () { return this._maxY; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "iteration", {
        get: function () { return this._iteration; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "numberOfPoints", {
        get: function () { return this.numPoints; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "numberOfPointsPerRow", {
        get: function () { return this.pointsPerRow; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "numberOfRows", {
        get: function () { return this.numRows; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "embeddingCoordinates", {
        get: function () { return this.embedding; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "embedding2D", {
        get: function () {
            var _this = this;
            var result = tf.tidy(function () {
                var reshaped = _this.embedding.reshape([_this.numRows * _this.pointsPerRow, 2])
                    .slice([0, 0], [_this.numPoints, 2]);
                return reshaped;
            });
            return result;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "embeddingTexture", {
        get: function () {
            return this.backend.getTexture(this.embedding.dataId);
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "splatTexture", {
        get: function () { return this._splatTexture; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "normalizationQ", {
        get: function () { return this._normQ; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "exaggerationAtCurrentIteration", {
        get: function () {
            return this._exaggeration.get();
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "exaggeration", {
        get: function () {
            return this.rawExaggeration;
        },
        set: function (ex) {
            this.rawExaggeration = ex;
            if (typeof ex === 'number') {
                if (ex < 1) {
                    throw Error('Exaggeration must be greater then or equal to one');
                }
            }
            else {
                for (var i = 0; i < ex.length; ++i) {
                    if (ex[i].value < 1) {
                        throw Error('Exaggeration must be greater then or equal to one');
                    }
                    if (ex[i].iteration < 0) {
                        throw Error('Piecewise linear exaggeration function \
                                        must have poistive iteration values');
                    }
                }
                for (var i = 0; i < ex.length - 1; ++i) {
                    if (ex[i].iteration >= ex[i + 1].iteration) {
                        throw Error('Piecewise linear exaggeration function \
                                      must have increasing iteration values');
                    }
                }
                if (ex.length === 1) {
                    this.exaggeration = ex[0].value;
                }
            }
            this.updateExaggeration();
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "momentum", {
        get: function () { return this._momentum.get(); },
        set: function (mom) {
            if (mom < 0 || mom > 1) {
                throw Error('Momentum must be in the [0,1] range');
            }
            this._momentum.dispose();
            this._momentum = tf.scalar(mom);
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TSNEOptimizer.prototype, "eta", {
        get: function () { return this._eta; },
        set: function (eta) {
            if (eta <= 0) {
                throw Error('ETA must be greater then zero');
            }
            this._eta = eta;
        },
        enumerable: true,
        configurable: true
    });
    TSNEOptimizer.prototype.dispose = function () {
        this.embedding.dispose();
        this.gradient.dispose();
        this._momentum.dispose();
        this._exaggeration.dispose();
        this.gpgpu.gl.deleteTexture(this._splatTexture);
        this.gpgpu.gl.deleteTexture(this.kernelTexture);
        if (this.kernelTexture != null) {
            this.gpgpu.gl.deleteTexture(this.probOffsetTexture);
        }
        if (this.kernelTexture != null) {
            this.gpgpu.gl.deleteTexture(this.probNeighIdTexture);
        }
        if (this.kernelTexture != null) {
            this.gpgpu.gl.deleteTexture(this.probTexture);
        }
        this.gpgpu.gl.deleteBuffer(this.splatVertexIdBuffer);
        this.gpgpu.gl.deleteProgram(this.embeddingInitializationProgram);
        this.gpgpu.gl.deleteProgram(this.embeddingSplatterProgram);
        this.gpgpu.gl.deleteProgram(this.qInterpolatorProgram);
        this.gpgpu.gl.deleteProgram(this.xyInterpolatorProgram);
        this.gpgpu.gl.deleteProgram(this.attractiveForcesProgram);
        this.gpgpu.gl.deleteProgram(this.distributionParameterssComputationProgram);
        this.gpgpu.gl.deleteProgram(this.gaussiaDistributionsFromDistancesProgram);
    };
    TSNEOptimizer.prototype.initializeEmbedding = function () {
        var _this = this;
        if (this.embedding != null) {
            this.embedding.dispose();
        }
        if (this.gradient != null) {
            this.gradient.dispose();
        }
        this.gradient = tf.zeros([this.numRows, this.pointsPerRow * 2]);
        this.embedding = tf.tidy(function () {
            var randomData = tf.randomUniform([_this.numRows, _this.pointsPerRow * 2]);
            var embedding = tf.zeros([_this.numRows, _this.pointsPerRow * 2]);
            _this.initializeEmbeddingPositions(embedding, randomData);
            return embedding;
        });
        var maxEmbeddingAbsCoordinate = 3;
        this._minX = -maxEmbeddingAbsCoordinate;
        this._minY = -maxEmbeddingAbsCoordinate;
        this._maxX = maxEmbeddingAbsCoordinate;
        this._maxY = maxEmbeddingAbsCoordinate;
        this.log('\tmin X', this._minX);
        this.log('\tmax X', this._maxX);
        this.log('\tmin Y', this._minY);
        this.log('\tmax Y', this._maxY);
        this._iteration = 0;
    };
    TSNEOptimizer.prototype.initializeNeighbors = function (numNeighPerRow, offsets, probabilities, neighIds) {
        this.numNeighPerRow = numNeighPerRow;
        this.probOffsetTexture = offsets;
        this.probTexture = probabilities;
        this.probNeighIdTexture = neighIds;
    };
    TSNEOptimizer.prototype.initializeNeighborsFromKNNGraph = function (numPoints, numNeighbors, distances, indices) {
        return __awaiter(this, void 0, void 0, function () {
            var pointsPerRow, numRows, dataShape, textureValues, i, n, id, knnGraphTexture;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        pointsPerRow = Math.max(1, Math.floor(Math.sqrt(numPoints * numNeighbors) / numNeighbors));
                        numRows = Math.ceil(numPoints / pointsPerRow);
                        dataShape = { numPoints: numPoints, pixelsPerPoint: numNeighbors, numRows: numRows, pointsPerRow: pointsPerRow };
                        textureValues = new Float32Array(pointsPerRow * numNeighbors * numRows * 2);
                        for (i = 0; i < numPoints; ++i) {
                            for (n = 0; n < numNeighbors; ++n) {
                                id = (i * numNeighbors + n);
                                textureValues[id * 2] = indices[id];
                                textureValues[id * 2 + 1] = distances[id];
                            }
                        }
                        knnGraphTexture = gl_util.createAndConfigureTexture(this.gpgpu.gl, pointsPerRow * numNeighbors, numRows, 2, textureValues);
                        return [4, this.initializeNeighborsFromKNNTexture(dataShape, knnGraphTexture)];
                    case 1:
                        _a.sent();
                        this.gpgpu.gl.deleteTexture(knnGraphTexture);
                        return [2];
                }
            });
        });
    };
    TSNEOptimizer.prototype.initializeNeighborsFromKNNTexture = function (shape, knnGraph) {
        return __awaiter(this, void 0, void 0, function () {
            var distParamTexture, gaussianDistributions, perplexity, gaussianDistributionsData, knnIndices, copyIndicesProgram, knnIndicesData, asymNeighIds, i, d, linearId, neighborCounter, neighborLinearOffset, i, i, check, maxValue, maxId, i, offsets, pointOffset, i, totalNeighbors, probabilities, neighIds, assignedNeighborCounter, i, n, linearId, pointId, probability, symMatrixDirectId, symMatrixIndirectId;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        this.log('Asymmetric neighborhood initialization...');
                        if (shape.numPoints !== this.numPoints) {
                            throw new Error("KNN size and number of points must agree" +
                                ("(" + shape.numPoints + "," + this.numPoints + ")"));
                        }
                        this.log("Create distribution params texture: \n        " + shape.pointsPerRow + " x " + shape.numRows);
                        distParamTexture = gl_util.createAndConfigureTexture(this.gpgpu.gl, shape.pointsPerRow, shape.numRows, 2);
                        this.log('Create zeroed distribution tensor');
                        gaussianDistributions = tf.zeros([shape.numRows, shape.pointsPerRow * shape.pixelsPerPoint]);
                        perplexity = shape.pixelsPerPoint / 3;
                        this.gpgpu.enableAutomaticDebugValidation(true);
                        this.log('Computing distribution params');
                        this.computeDistributionParameters(distParamTexture, shape, perplexity, knnGraph);
                        this.log('Computing Gaussian distn');
                        this.computeGaussianDistributions(gaussianDistributions, distParamTexture, shape, knnGraph);
                        this.log('Retrieve Gaussian distn');
                        gaussianDistributionsData = gaussianDistributions.dataSync();
                        knnIndices = tf.zeros([shape.numRows, shape.pointsPerRow * shape.pixelsPerPoint]);
                        this.log('Create copy indices program', knnIndices.shape);
                        copyIndicesProgram = knn_util.createCopyIndicesProgram(this.gpgpu);
                        this.log('Execute copy indices program', knnIndices.shape);
                        knn_util.executeCopyIndicesProgram(this.gpgpu, copyIndicesProgram, knnGraph, shape, this.backend.getTexture(knnIndices.dataId));
                        return [4, knnIndices.data()];
                    case 1:
                        knnIndicesData = _a.sent();
                        this.log('knn Indices', knnIndices);
                        asymNeighIds = new Float32Array(shape.numPoints * shape.pixelsPerPoint);
                        for (i = 0; i < this.numPoints; ++i) {
                            for (d = 0; d < shape.pixelsPerPoint; ++d) {
                                linearId = i * shape.pixelsPerPoint + d;
                                asymNeighIds[i * shape.pixelsPerPoint + d] = knnIndicesData[linearId];
                            }
                        }
                        this.log('NeighIds', asymNeighIds);
                        neighborCounter = new Uint32Array(this.numPoints);
                        neighborLinearOffset = new Uint32Array(this.numPoints);
                        for (i = 0; i < shape.numPoints * shape.pixelsPerPoint; ++i) {
                            ++neighborCounter[asymNeighIds[i]];
                        }
                        for (i = 1; i < shape.numPoints; ++i) {
                            neighborLinearOffset[i] = neighborLinearOffset[i - 1] +
                                neighborCounter[i - 1] + shape.pixelsPerPoint;
                        }
                        this.log('Counter', neighborCounter);
                        this.log('Linear offset', neighborLinearOffset);
                        check = 0;
                        maxValue = 0;
                        maxId = 0;
                        for (i = 0; i < neighborCounter.length; ++i) {
                            check += neighborCounter[i];
                            if (neighborCounter[i] > maxValue) {
                                maxValue = neighborCounter[i];
                                maxId = i;
                            }
                        }
                        this.log('Number of indirect links', check);
                        this.log('Most central point', maxId);
                        this.log('Number of indirect links for the central point', maxValue);
                        this.numNeighPerRow =
                            Math.ceil(Math.sqrt(shape.numPoints * shape.pixelsPerPoint * 2));
                        this.log('numNeighPerRow', this.numNeighPerRow);
                        {
                            offsets = new Float32Array(this.pointsPerRow * this.numRows * 3);
                            pointOffset = 0;
                            for (i = 0; i < this.numPoints; ++i) {
                                totalNeighbors = shape.pixelsPerPoint + neighborCounter[i];
                                offsets[i * 3 + 0] = (pointOffset) % (this.numNeighPerRow);
                                offsets[i * 3 + 1] = Math.floor((pointOffset) / (this.numNeighPerRow));
                                offsets[i * 3 + 2] = totalNeighbors;
                                pointOffset += totalNeighbors;
                            }
                            this.log('Offsets', offsets);
                            this.probOffsetTexture = gl_util.createAndConfigureTexture(this.gpgpu.gl, this.pointsPerRow, this.numRows, 3, offsets);
                        }
                        {
                            probabilities = new Float32Array(this.numNeighPerRow * this.numNeighPerRow);
                            neighIds = new Float32Array(this.numNeighPerRow * this.numNeighPerRow);
                            assignedNeighborCounter = new Uint32Array(this.numPoints);
                            for (i = 0; i < this.numPoints; ++i) {
                                for (n = 0; n < shape.pixelsPerPoint; ++n) {
                                    linearId = i * shape.pixelsPerPoint + n;
                                    pointId = knnIndicesData[linearId];
                                    probability = gaussianDistributionsData[linearId];
                                    symMatrixDirectId = neighborLinearOffset[i] + n;
                                    symMatrixIndirectId = neighborLinearOffset[pointId] +
                                        shape.pixelsPerPoint +
                                        assignedNeighborCounter[pointId];
                                    probabilities[symMatrixDirectId] = probability;
                                    probabilities[symMatrixIndirectId] = probability;
                                    neighIds[symMatrixDirectId] = pointId;
                                    neighIds[symMatrixIndirectId] = i;
                                    ++assignedNeighborCounter[pointId];
                                }
                            }
                            this.probTexture = gl_util.createAndConfigureTexture(this.gpgpu.gl, this.numNeighPerRow, this.numNeighPerRow, 1, probabilities);
                            this.probNeighIdTexture = gl_util.createAndConfigureTexture(this.gpgpu.gl, this.numNeighPerRow, this.numNeighPerRow, 1, neighIds);
                        }
                        gaussianDistributions.dispose();
                        knnIndices.dispose();
                        this.log('...done!');
                        return [2];
                }
            });
        });
    };
    TSNEOptimizer.prototype.initializedNeighborhoods = function () {
        return this.probNeighIdTexture != null;
    };
    TSNEOptimizer.prototype.updateExaggeration = function () {
        if (this._exaggeration !== undefined) {
            this._exaggeration.dispose();
        }
        if (typeof this.rawExaggeration === 'number') {
            this._exaggeration = tf.scalar(this.rawExaggeration);
            return;
        }
        if (this._iteration <= this.rawExaggeration[0].iteration) {
            this._exaggeration = tf.scalar(this.rawExaggeration[0].value);
            return;
        }
        if (this._iteration >=
            this.rawExaggeration[this.rawExaggeration.length - 1].iteration) {
            this._exaggeration = tf.scalar(this.rawExaggeration[this.rawExaggeration.length - 1].value);
            return;
        }
        var i = 0;
        while (i < this.rawExaggeration.length &&
            this._iteration < this.rawExaggeration[i].iteration) {
            ++i;
        }
        var it0 = this.rawExaggeration[i].iteration;
        var it1 = this.rawExaggeration[i + 1].iteration;
        var v0 = this.rawExaggeration[i].value;
        var v1 = this.rawExaggeration[i + 1].value;
        var f = (it1 - this._iteration) / (it1 - it0);
        var v = v0 * f + v1 * (1 - f);
        this._exaggeration = tf.scalar(v);
    };
    TSNEOptimizer.prototype.iterate = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, normQ, _b;
            var _this = this;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        if (!this.initializedNeighborhoods()) {
                            throw new Error('No neighborhoods defined. You may want to call\
                    initializeNeighbors or initializeNeighborsFromKNNGraph');
                        }
                        this.updateSplatTextureDiameter();
                        this.updateExaggeration();
                        _a = __read(tf.tidy(function () {
                            _this.splatPoints();
                            var interpQ = tf.zeros([_this.numRows, _this.pointsPerRow]);
                            var interpXY = tf.zeros([_this.numRows, _this.pointsPerRow * 2]);
                            _this.computeInterpolatedQ(interpQ);
                            _this.computeInterpolatedXY(interpXY);
                            var normQ = interpQ.sum();
                            var repulsiveForces = interpXY.div(tf.tensor(normQ.dataSync(), normQ.shape));
                            var attractiveForces = tf.zeros([_this.numRows, _this.pointsPerRow * 2]);
                            _this.computeAttractiveForces(attractiveForces);
                            var gradientIter = attractiveForces.mul(_this._exaggeration).sub(repulsiveForces);
                            var gradient = _this.gradient.mul(_this._momentum).sub(gradientIter);
                            _this.gradient.dispose();
                            return [gradient, normQ];
                        }), 2), this.gradient = _a[0], normQ = _a[1];
                        _b = this;
                        return [4, normQ.data()];
                    case 1:
                        _b._normQ = (_c.sent())[0];
                        normQ.dispose();
                        this.embedding = tf.tidy(function () {
                            var embedding = tf.add(tf.tensor(_this.embedding.dataSync(), _this.embedding.shape), tf.tensor(_this.gradient.dataSync(), _this.gradient.shape));
                            _this.embedding.dispose();
                            return embedding;
                        });
                        return [4, this.computeBoundaries()];
                    case 2:
                        _c.sent();
                        ++this._iteration;
                        return [2];
                }
            });
        });
    };
    TSNEOptimizer.prototype.log = function (str, obj) {
        if (this.verbose) {
            if (obj != null) {
                console.log(str + ": \t" + obj);
            }
            else {
                console.log(str);
            }
        }
    };
    TSNEOptimizer.prototype.initializeRepulsiveForceTextures = function () {
        this._splatTexture = gl_util.createAndConfigureInterpolatedTexture(this.gpgpu.gl, this.splatTextureDiameter, this.splatTextureDiameter, 4, null);
        this.kernelSupport = 2.5;
        var kernel = new Float32Array(this.kernelTextureDiameter *
            this.kernelTextureDiameter * 4);
        var kernelRadius = Math.floor(this.kernelTextureDiameter / 2);
        var j = 0;
        var i = 0;
        for (j = 0; j < this.kernelTextureDiameter; ++j) {
            for (i = 0; i < this.kernelTextureDiameter; ++i) {
                var x = (i - kernelRadius) / kernelRadius * this.kernelSupport;
                var y = (j - kernelRadius) / kernelRadius * this.kernelSupport;
                var euclSquared = x * x + y * y;
                var tStudent = 1. / (1. + euclSquared);
                var id = (j * this.kernelTextureDiameter + i) * 4;
                kernel[id + 0] = tStudent;
                kernel[id + 1] = tStudent * tStudent * x;
                kernel[id + 2] = tStudent * tStudent * y;
                kernel[id + 3] = 1;
            }
        }
        this.kernelTexture = gl_util.createAndConfigureInterpolatedTexture(this.gpgpu.gl, this.kernelTextureDiameter, this.kernelTextureDiameter, 4, kernel);
    };
    TSNEOptimizer.prototype.initilizeCustomWebGLPrograms = function () {
        this.log('\tCreating custom programs...');
        this.embeddingInitializationProgram =
            tsne_util.createEmbeddingInitializationProgram(this.gpgpu);
        this.embeddingSplatterProgram =
            tsne_util.createEmbeddingSplatterProgram(this.gpgpu);
        var splatVertexId = new Float32Array(this.numPoints * 6);
        {
            var i = 0;
            var id = 0;
            for (i = 0; i < this.numPoints; ++i) {
                id = i * 6;
                splatVertexId[id + 0] = 0 + i * 4;
                splatVertexId[id + 1] = 1 + i * 4;
                splatVertexId[id + 2] = 2 + i * 4;
                splatVertexId[id + 3] = 0 + i * 4;
                splatVertexId[id + 4] = 2 + i * 4;
                splatVertexId[id + 5] = 3 + i * 4;
            }
        }
        this.splatVertexIdBuffer = tf.webgl.webgl_util.createStaticVertexBuffer(this.gpgpu.gl, splatVertexId);
        this.qInterpolatorProgram =
            tsne_util.createQInterpolatorProgram(this.gpgpu);
        this.xyInterpolatorProgram =
            tsne_util.createXYInterpolatorProgram(this.gpgpu);
        this.attractiveForcesProgram =
            tsne_util.createAttractiveForcesComputationProgram(this.gpgpu);
        this.distributionParameterssComputationProgram =
            tsne_util.createDistributionParametersComputationProgram(this.gpgpu);
        this.gaussiaDistributionsFromDistancesProgram =
            tsne_util.createGaussiaDistributionsFromDistancesProgram(this.gpgpu);
    };
    TSNEOptimizer.prototype.computeBoundaries = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, min, max, minData, maxData, percentageOffset, offsetX, offsetY;
            var _this = this;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = __read(tf.tidy(function () {
                            var embedding2D = _this.embedding.reshape([_this.numRows * _this.pointsPerRow, 2])
                                .slice([0, 0], [_this.numPoints, 2]);
                            var min = embedding2D.min(0);
                            var max = embedding2D.max(0);
                            return [min, max];
                        }), 2), min = _a[0], max = _a[1];
                        return [4, min.data()];
                    case 1:
                        minData = _b.sent();
                        return [4, max.data()];
                    case 2:
                        maxData = _b.sent();
                        percentageOffset = 0.05;
                        offsetX = (maxData[0] - minData[0]) * percentageOffset;
                        this._minX = minData[0] - offsetX;
                        this._maxX = maxData[0] + offsetX;
                        offsetY = (maxData[1] - minData[1]) * percentageOffset;
                        this._minY = minData[1] - offsetY;
                        this._maxY = maxData[1] + offsetY;
                        min.dispose();
                        max.dispose();
                        return [2];
                }
            });
        });
    };
    TSNEOptimizer.prototype.updateSplatTextureDiameter = function () {
        var maxSpace = Math.max(this._maxX - this._minX, this._maxY - this._minY);
        var spacePerPixel = 0.35;
        var maxTextureDiameter = 5000;
        var textureDiameter = Math.min(Math.ceil(Math.max(maxSpace / spacePerPixel, 5)), maxTextureDiameter);
        var percChange = Math.abs(this.splatTextureDiameter - textureDiameter) /
            this.splatTextureDiameter;
        if (percChange >= 0.2) {
            this.log('Updating splat-texture diameter', textureDiameter);
            this.gpgpu.gl.deleteTexture(this._splatTexture);
            this.splatTextureDiameter = textureDiameter;
            this._splatTexture = gl_util.createAndConfigureInterpolatedTexture(this.gpgpu.gl, this.splatTextureDiameter, this.splatTextureDiameter, 4, null);
        }
    };
    TSNEOptimizer.prototype.initializeEmbeddingPositions = function (embedding, random) {
        tsne_util.executeEmbeddingInitializationProgram(this.gpgpu, this.embeddingInitializationProgram, this.backend.getTexture(random.dataId), this.numPoints, this.pointsPerRow, this.numRows, this.backend.getTexture(embedding.dataId));
    };
    TSNEOptimizer.prototype.splatPoints = function () {
        tsne_util.executeEmbeddingSplatterProgram(this.gpgpu, this.embeddingSplatterProgram, this._splatTexture, this.backend.getTexture(this.embedding.dataId), this.kernelTexture, this.splatTextureDiameter, this.numPoints, this._minX, this._minY, this._maxX, this._maxY, this.kernelSupport, this.pointsPerRow, this.numRows, this.splatVertexIdBuffer);
    };
    TSNEOptimizer.prototype.computeInterpolatedQ = function (interpolatedQ) {
        tsne_util.executeQInterpolatorProgram(this.gpgpu, this.qInterpolatorProgram, this._splatTexture, this.backend.getTexture(this.embedding.dataId), this.numPoints, this._minX, this._minY, this._maxX, this._maxY, this.pointsPerRow, this.numRows, this.backend.getTexture(interpolatedQ.dataId));
    };
    TSNEOptimizer.prototype.computeInterpolatedXY = function (interpolatedXY) {
        tsne_util.executeXYInterpolatorProgram(this.gpgpu, this.xyInterpolatorProgram, this._splatTexture, this.backend.getTexture(this.embedding.dataId), this.backend.getTexture(interpolatedXY.dataId), this.numPoints, this._minX, this._minY, this._maxX, this._maxY, this.pointsPerRow, this.numRows, this._eta);
    };
    TSNEOptimizer.prototype.computeAttractiveForces = function (attractiveForces) {
        tsne_util.executeAttractiveForcesComputationProgram(this.gpgpu, this.attractiveForcesProgram, this.backend.getTexture(this.embedding.dataId), this.probOffsetTexture, this.probNeighIdTexture, this.probTexture, this.numPoints, this.numNeighPerRow, this.pointsPerRow, this.numRows, this._eta, this.backend.getTexture(attractiveForces.dataId));
    };
    TSNEOptimizer.prototype.computeDistributionParameters = function (distributionParameters, shape, perplexity, knnGraph) {
        try {
            tsne_util.executeDistributionParametersComputationProgram(this.gpgpu, this.distributionParameterssComputationProgram, knnGraph, shape.numPoints, shape.pixelsPerPoint, shape.pointsPerRow, shape.numRows, perplexity, distributionParameters);
        }
        catch (e) {
            console.log('Error in executeDistributionParametersComputationProgram' +
                e.toString());
        }
    };
    TSNEOptimizer.prototype.computeGaussianDistributions = function (distributions, distributionParameters, shape, knnGraph) {
        try {
            var distTexture = this.backend.getTexture(distributions.dataId);
            tsne_util.executeGaussiaDistributionsFromDistancesProgram(this.gpgpu, this.gaussiaDistributionsFromDistancesProgram, knnGraph, distributionParameters, shape.numPoints, shape.pixelsPerPoint, shape.pointsPerRow, shape.numRows, distTexture);
        }
        catch (e) {
            console.log('Error in executeGaussiaDistributionsFromDistancesProgram' +
                e.toString());
        }
    };
    return TSNEOptimizer;
}());
exports.TSNEOptimizer = TSNEOptimizer;
//# sourceMappingURL=tsne_optimizer.js.map