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
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var knn_1 = require("./knn");
var tensor_to_data_texture_1 = require("./tensor_to_data_texture");
var tsne_optimizer_1 = require("./tsne_optimizer");
function maximumPerplexity() {
    var backend = tf.ENV.findBackend('webgl');
    if (backend === null) {
        throw Error('WebGL backend is not available');
    }
    var gl = backend.getGPGPUContext().gl;
    var maxVaryingVectors = gl.getParameter(gl.MAX_VARYING_VECTORS);
    var numNeighbors = (maxVaryingVectors - 1) * 4;
    var maximumPerplexity = Math.floor(numNeighbors / 3);
    return maximumPerplexity;
}
exports.maximumPerplexity = maximumPerplexity;
function tsne(data, config) {
    return new TSNE(data, config);
}
exports.tsne = tsne;
var TSNE = (function () {
    function TSNE(data, config) {
        this.initialized = false;
        this.probabilitiesInitialized = false;
        this.data = data;
        this.config = config;
        var inputShape = this.data.shape;
        this.numPoints = inputShape[0];
        this.numDimensions = inputShape[1];
        if (inputShape.length !== 2) {
            throw Error('computeTSNE: input tensor must be 2-dimensional');
        }
        var perplexity = 18;
        if (this.config !== undefined) {
            if (this.config.perplexity !== undefined) {
                perplexity = this.config.perplexity;
            }
        }
        var maxPerplexity = maximumPerplexity();
        if (perplexity > maxPerplexity) {
            throw Error("computeTSNE: perplexity cannot be greater than" +
                (maxPerplexity + " on this machine"));
        }
    }
    TSNE.prototype.initialize = function () {
        return __awaiter(this, void 0, void 0, function () {
            var perplexity, exaggeration, exaggerationIter, exaggerationDecayIter, momentum, _a, exaggerationPolyline, maximumEta, minimumEta, numPointsMaximumEta;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        perplexity = 18;
                        exaggeration = 4;
                        exaggerationIter = 300;
                        exaggerationDecayIter = 200;
                        momentum = 0.8;
                        this.verbose = false;
                        this.knnMode = 'auto';
                        if (this.config !== undefined) {
                            if (this.config.perplexity !== undefined) {
                                perplexity = this.config.perplexity;
                            }
                            if (this.config.exaggeration !== undefined) {
                                exaggeration = this.config.exaggeration;
                            }
                            if (this.config.exaggerationIter !== undefined) {
                                exaggerationIter = this.config.exaggerationIter;
                            }
                            if (this.config.exaggerationDecayIter !== undefined) {
                                exaggerationDecayIter = this.config.exaggerationDecayIter;
                            }
                            if (this.config.momentum !== undefined) {
                                momentum = this.config.momentum;
                            }
                            if (this.config.verbose !== undefined) {
                                this.verbose = this.config.verbose;
                            }
                            if (this.config.knnMode !== undefined) {
                                this.knnMode = this.config.knnMode;
                            }
                        }
                        this.numNeighbors = Math.floor((perplexity * 3) / 4) * 4;
                        _a = this;
                        return [4, tensor_to_data_texture_1.tensorToDataTexture(this.data)];
                    case 1:
                        _a.packedData = _b.sent();
                        if (this.verbose) {
                            console.log("Number of points:\t" + this.numPoints);
                            console.log("Number of dimensions:\t " + this.numDimensions);
                            console.log("Number of neighbors:\t" + this.numNeighbors);
                            console.log("kNN mode:\t" + this.knnMode);
                        }
                        this.knnEstimator = new knn_1.KNNEstimator(this.packedData.texture, this.packedData.shape, this.numPoints, this.numDimensions, this.numNeighbors, this.verbose);
                        this.optimizer = new tsne_optimizer_1.TSNEOptimizer(this.numPoints, this.verbose);
                        exaggerationPolyline = [
                            { iteration: exaggerationIter, value: exaggeration },
                            { iteration: exaggerationIter + exaggerationDecayIter, value: 1 }
                        ];
                        if (this.verbose) {
                            console.log("Exaggerating for " + exaggerationPolyline[0].iteration + " " +
                                ("iterations with a value of " + exaggerationPolyline[0].value + ". ") +
                                ("Exaggeration is removed after " + exaggerationPolyline[1].iteration + "."));
                        }
                        this.optimizer.exaggeration = exaggerationPolyline;
                        this.optimizer.momentum = momentum;
                        maximumEta = 2500;
                        minimumEta = 250;
                        numPointsMaximumEta = 2000;
                        if (this.numPoints > numPointsMaximumEta) {
                            this.optimizer.eta = maximumEta;
                        }
                        else {
                            this.optimizer.eta = minimumEta +
                                (maximumEta - minimumEta) * (this.numPoints / numPointsMaximumEta);
                        }
                        this.initialized = true;
                        return [2];
                }
            });
        });
    };
    TSNE.prototype.compute = function (iterations) {
        if (iterations === void 0) { iterations = 1000; }
        return __awaiter(this, void 0, void 0, function () {
            var knnIter;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        knnIter = this.knnIterations();
                        if (this.verbose) {
                            console.log("Number of KNN iterations:\t" + knnIter);
                            console.log('Computing the KNN...');
                        }
                        return [4, this.iterateKnn(knnIter)];
                    case 1:
                        _a.sent();
                        if (this.verbose) {
                            console.log('Computing the tSNE embedding...');
                        }
                        return [4, this.iterate(iterations)];
                    case 2:
                        _a.sent();
                        if (this.verbose) {
                            console.log('Done!');
                        }
                        return [2];
                }
            });
        });
    };
    TSNE.prototype.iterateKnn = function (iterations) {
        if (iterations === void 0) { iterations = 1; }
        return __awaiter(this, void 0, void 0, function () {
            var iter, syncCounter;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!!this.initialized) return [3, 2];
                        return [4, this.initialize()];
                    case 1:
                        _a.sent();
                        _a.label = 2;
                    case 2:
                        this.probabilitiesInitialized = false;
                        iter = 0;
                        _a.label = 3;
                    case 3:
                        if (!(iter < iterations)) return [3, 6];
                        this.knnEstimator.iterateKNNDescent();
                        syncCounter = 10;
                        if ((this.knnEstimator.iteration % 100) === 0 && this.verbose) {
                            console.log("Iteration KNN:\t" + this.knnEstimator.iteration);
                        }
                        if (!(this.knnEstimator.iteration % syncCounter === 0)) return [3, 5];
                        return [4, this.knnEstimator.forceFlush()];
                    case 4:
                        _a.sent();
                        _a.label = 5;
                    case 5:
                        ++iter;
                        return [3, 3];
                    case 6: return [2];
                }
            });
        });
    };
    TSNE.prototype.iterate = function (iterations) {
        if (iterations === void 0) { iterations = 1; }
        return __awaiter(this, void 0, void 0, function () {
            var iter;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!!this.probabilitiesInitialized) return [3, 2];
                        return [4, this.initializeProbabilities()];
                    case 1:
                        _a.sent();
                        _a.label = 2;
                    case 2:
                        iter = 0;
                        _a.label = 3;
                    case 3:
                        if (!(iter < iterations)) return [3, 6];
                        return [4, this.optimizer.iterate()];
                    case 4:
                        _a.sent();
                        if ((this.optimizer.iteration % 100) === 0 && this.verbose) {
                            console.log("Iteration tSNE:\t" + this.optimizer.iteration);
                        }
                        _a.label = 5;
                    case 5:
                        ++iter;
                        return [3, 3];
                    case 6: return [2];
                }
            });
        });
    };
    TSNE.prototype.knnIterations = function () {
        return Math.ceil(this.numPoints / 20);
    };
    TSNE.prototype.coordinates = function (normalized) {
        var _this = this;
        if (normalized === void 0) { normalized = true; }
        if (normalized) {
            return tf.tidy(function () {
                var rangeX = _this.optimizer.maxX - _this.optimizer.minX;
                var rangeY = _this.optimizer.maxY - _this.optimizer.minY;
                var min = tf.tensor2d([_this.optimizer.minX, _this.optimizer.minY], [1, 2]);
                var max = tf.tensor2d([_this.optimizer.maxX, _this.optimizer.maxY], [1, 2]);
                var range = max.sub(min);
                var maxRange = tf.max(tf.tensor(range.dataSync()));
                var offset = tf.tidy(function () {
                    if (rangeX < rangeY) {
                        return tf.tensor2d([(rangeY - rangeX) / 2, 0], [1, 2]);
                    }
                    else {
                        return tf.tensor2d([0, (rangeX - rangeY) / 2], [1, 2]);
                    }
                });
                return _this.optimizer.embedding2D.sub(min).add(offset).div(maxRange);
            });
        }
        else {
            return this.optimizer.embedding2D;
        }
    };
    TSNE.prototype.coordsArray = function (normalized) {
        if (normalized === void 0) { normalized = true; }
        return __awaiter(this, void 0, void 0, function () {
            var coordsData, coords, i;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, this.coordinates(normalized).data()];
                    case 1:
                        coordsData = _a.sent();
                        coords = [];
                        for (i = 0; i < coordsData.length; i += 2) {
                            coords.push([coordsData[i], coordsData[i + 1]]);
                        }
                        return [2, coords];
                }
            });
        });
    };
    TSNE.prototype.knnTotalDistance = function () {
        return __awaiter(this, void 0, void 0, function () {
            var sum;
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        sum = tf.tidy(function () {
                            var distanceTensor = _this.knnEstimator.distancesTensor();
                            return distanceTensor.sum();
                        });
                        return [4, sum.data()];
                    case 1: return [2, (_a.sent())[0]];
                }
            });
        });
    };
    TSNE.prototype.initializeProbabilities = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (this.verbose) {
                            console.log("Initializing probabilities");
                        }
                        return [4, this.optimizer.initializeNeighborsFromKNNTexture(this.knnEstimator.knnShape, this.knnEstimator.knn())];
                    case 1:
                        _a.sent();
                        if (this.verbose) {
                            console.log("Initialized probabilities from kNN Texture");
                        }
                        this.probabilitiesInitialized = true;
                        return [2];
                }
            });
        });
    };
    return TSNE;
}());
exports.TSNE = TSNE;
//# sourceMappingURL=tsne.js.map