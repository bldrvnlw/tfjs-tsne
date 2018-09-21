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
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var dataset_util = require("./dataset_util");
var tf_tsne = require("./tsne_optimizer");
describe('TSNEOptimizer class', function () {
    it('is initialized correctly', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(tsne.minX).toBeLessThan(tsne.maxX);
        expect(tsne.minY).toBeLessThan(tsne.maxY);
        expect(tsne.numberOfPoints).toBe(100);
        expect(tsne.numberOfPointsPerRow).toBe(8);
        expect(tsne.numberOfRows).toBe(13);
        tsne.dispose();
    });
    it('requires the neighborhoods to perform iterations', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, e_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(100, false);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, tsne.iterate()];
                case 2:
                    _a.sent();
                    return [3, 4];
                case 3:
                    e_1 = _a.sent();
                    expect(true);
                    tsne.dispose();
                    return [2];
                case 4:
                    fail('Method did not throw an exception');
                    return [2];
            }
        });
    }); });
    it('returns a properly sized embedding', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var embedding2D = tsne.embedding2D;
        expect(embedding2D.shape[0]).toBe(100);
        expect(embedding2D.shape[1]).toBe(2);
        embedding2D.dispose();
        tsne.dispose();
    });
    it('detects a mismatched data shape', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, knnGraph, e_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(100, false);
                    knnGraph = dataset_util.generateKNNClusterData(1000, 10, 100);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, tsne.initializeNeighborsFromKNNGraph(1000, 100, knnGraph.distances, knnGraph.indices)];
                case 2:
                    _a.sent();
                    return [3, 4];
                case 3:
                    e_2 = _a.sent();
                    expect(true);
                    tsne.dispose();
                    return [2];
                case 4:
                    fail('Method did not throw an exception');
                    return [2];
            }
        });
    }); });
    it('requires 4 tensors', function () {
        var tsne = new tf_tsne.TSNEOptimizer(1000, false);
        expect(tf.memory().numTensors).toBe(4);
        tsne.dispose();
    });
    it('disposes its tensors', function () {
        var tsne = new tf_tsne.TSNEOptimizer(1000, false);
        tsne.dispose();
        expect(tf.memory().numTensors).toBe(0);
    });
    it('keeps the number of tensors constant during neighbors initialization', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, knnGraph;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(1000, false);
                    knnGraph = dataset_util.generateKNNClusterData(1000, 10, 100);
                    return [4, tsne.initializeNeighborsFromKNNGraph(1000, 100, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    expect(tf.memory().numTensors).toBe(4);
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
    it('keeps the number of tensors constant during SGD', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, knnGraph, numTensors, numIter, i;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(1000, false);
                    knnGraph = dataset_util.generateKNNClusterData(1000, 10, 30);
                    return [4, tsne.initializeNeighborsFromKNNGraph(1000, 30, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    numTensors = tf.memory().numTensors;
                    numIter = 25;
                    i = 0;
                    _a.label = 2;
                case 2:
                    if (!(i < numIter)) return [3, 5];
                    return [4, tsne.iterate()];
                case 3:
                    _a.sent();
                    _a.label = 4;
                case 4:
                    ++i;
                    return [3, 2];
                case 5:
                    expect(tf.memory().numTensors).toBe(numTensors);
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
    it('initializes the iterations to 0', function () {
        var tsne = new tf_tsne.TSNEOptimizer(1000, false);
        expect(tsne.iteration).toBe(0);
        tsne.dispose();
    });
    it('counts the iterations', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, knnGraph, numIter, i;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(1000, false);
                    knnGraph = dataset_util.generateKNNClusterData(1000, 10, 30);
                    return [4, tsne.initializeNeighborsFromKNNGraph(1000, 30, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    numIter = 10;
                    i = 0;
                    _a.label = 2;
                case 2:
                    if (!(i < numIter)) return [3, 5];
                    return [4, tsne.iterate()];
                case 3:
                    _a.sent();
                    expect(tsne.iteration).toBe(i + 1);
                    _a.label = 4;
                case 4:
                    ++i;
                    return [3, 2];
                case 5:
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
    it('resets the iterations counter on re-init', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, knnGraph, numIter, i;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(1000, false);
                    knnGraph = dataset_util.generateKNNClusterData(1000, 10, 30);
                    return [4, tsne.initializeNeighborsFromKNNGraph(1000, 30, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    numIter = 10;
                    i = 0;
                    _a.label = 2;
                case 2:
                    if (!(i < numIter)) return [3, 5];
                    return [4, tsne.iterate()];
                case 3:
                    _a.sent();
                    _a.label = 4;
                case 4:
                    ++i;
                    return [3, 2];
                case 5:
                    tsne.initializeEmbedding();
                    expect(tsne.iteration).toBe(0);
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
    it('has proper ETA getter/setter', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(tsne.eta).toBeCloseTo(2500);
        tsne.eta = 1000;
        expect(tsne.eta).toBeCloseTo(1000);
        tsne.dispose();
    });
    it('has proper Momentum getter/setter', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(tsne.momentum).toBeCloseTo(0.8);
        tsne.momentum = 0.1;
        expect(tsne.momentum).toBeCloseTo(0.1);
        tsne.dispose();
    });
    it('has proper Exaggeration getter/setter', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(4);
        tsne.exaggeration = 3;
        expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
        tsne.dispose();
    });
    it('does not increase the tensor count when momentum is changed', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var numTensors = tf.memory().numTensors;
        tsne.momentum = 0.1;
        expect(tf.memory().numTensors).toBe(numTensors);
        tsne.dispose();
    });
    it('does not increase the tensor count when exaggeration is changed', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var numTensors = tf.memory().numTensors;
        tsne.exaggeration = 4;
        expect(tf.memory().numTensors).toBe(numTensors);
        tsne.dispose();
    });
    it('does not increase the tensor count after an embedding re-init', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var numTensors = tf.memory().numTensors;
        tsne.initializeEmbedding();
        expect(tf.memory().numTensors).toBe(numTensors);
        tsne.dispose();
    });
    it('throws if a negative momentum is set', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(function () {
            tsne.momentum = -0.1;
        }).toThrow();
        tsne.dispose();
    });
    it('does not throw if momentum is set to zero', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(function () {
            tsne.momentum = 0.;
        }).not.toThrow();
        tsne.dispose();
    });
    it('throws if a momentum higher than one is set', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(function () {
            tsne.momentum = 2;
        }).toThrow();
        tsne.dispose();
    });
    it('throws if exaggeration is set to a value lower than one', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(function () {
            tsne.exaggeration = 0.9;
        }).toThrow();
        tsne.dispose();
    });
    it('does not throw if exaggeration is set to one', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(function () {
            tsne.exaggeration = 1;
        }).not.toThrow();
        tsne.dispose();
    });
    it('accpets only piecewise linear exaggeration greater' +
        ' than or equal to 1 (0)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var ex = [{ iteration: 0, value: 2 }, { iteration: 100, value: 0.5 }];
        expect(function () {
            tsne.exaggeration = ex;
        }).toThrow();
        tsne.dispose();
    });
    it('accpets only piecewise linear exaggeration greater' +
        ' than or equal to 1 (1)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var ex = [{ iteration: 0, value: 0 }];
        expect(function () {
            tsne.exaggeration = ex;
        }).toThrow();
        tsne.dispose();
    });
    it('accpets exaggeration functions with non negative iteratiosn (0)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var ex = [{ iteration: 0, value: 2 }, { iteration: -100, value: 0.5 }];
        expect(function () {
            tsne.exaggeration = ex;
        }).toThrow();
        tsne.dispose();
    });
    it('accpets exaggeration functions with non negative iteratiosn (0)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var ex = [{ iteration: -100, value: 0.5 }];
        expect(function () {
            tsne.exaggeration = ex;
        }).toThrow();
        tsne.dispose();
    });
    it('accpets exaggeration functions (domain is always increasing 1)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var ex = [{ iteration: 200, value: 2 }, { iteration: 100, value: 1 }];
        expect(function () {
            tsne.exaggeration = ex;
        }).toThrow();
        tsne.dispose();
    });
    it('accpets exaggeration functions (domain is always increasing 2)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var ex = [{ iteration: 200, value: 2 }, { iteration: 200, value: 1 }];
        expect(function () {
            tsne.exaggeration = ex;
        }).toThrow();
        tsne.dispose();
    });
    it('throws if a non-positive ETA is set', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(function () {
            tsne.eta = -1000;
        }).toThrow();
        tsne.dispose();
    });
    it('throws if ETA is set to zero', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(function () {
            tsne.eta = 0;
        }).toThrow();
        tsne.dispose();
    });
    it('has proper exaggeration for the current iteration (0)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(4);
        tsne.exaggeration = 3;
        expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
        tsne.dispose();
    });
    it('has proper exaggeration for the current iteration (1)', function () {
        var tsne = new tf_tsne.TSNEOptimizer(100, false);
        var ex = [{ iteration: 0, value: 3 }];
        tsne.exaggeration = ex;
        expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
        tsne.dispose();
    });
    it('has proper exaggeration for the current iteration (1)', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, ex, knnGraph;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(100, false);
                    ex = [{ iteration: 0, value: 3 }];
                    tsne.exaggeration = ex;
                    knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
                    return [4, tsne.initializeNeighborsFromKNNGraph(100, 30, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    return [4, tsne.iterate()];
                case 2:
                    _a.sent();
                    return [4, tsne.iterate()];
                case 3:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
    it('has proper exaggeration for the current iteration (2)', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, ex, knnGraph;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(100, false);
                    ex = [{ iteration: 0, value: 3 }, { iteration: 2, value: 1 }];
                    tsne.exaggeration = ex;
                    knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
                    return [4, tsne.initializeNeighborsFromKNNGraph(100, 30, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    return [4, tsne.iterate()];
                case 2:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
                    return [4, tsne.iterate()];
                case 3:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(2);
                    return [4, tsne.iterate()];
                case 4:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
    it('has proper exaggeration for the current iteration (3)', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, ex, knnGraph;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(100, false);
                    ex = [{ iteration: 1, value: 3 }, { iteration: 3, value: 1 }];
                    tsne.exaggeration = ex;
                    knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
                    return [4, tsne.initializeNeighborsFromKNNGraph(100, 30, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    return [4, tsne.iterate()];
                case 2:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
                    return [4, tsne.iterate()];
                case 3:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
                    return [4, tsne.iterate()];
                case 4:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(2);
                    return [4, tsne.iterate()];
                case 5:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
                    return [4, tsne.iterate()];
                case 6:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
    it('has proper exaggeration for the current iteration (4)', function () { return __awaiter(_this, void 0, void 0, function () {
        var tsne, ex, knnGraph;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    tsne = new tf_tsne.TSNEOptimizer(100, false);
                    ex = [{ iteration: 0, value: 5 }, { iteration: 4, value: 1 }];
                    tsne.exaggeration = ex;
                    knnGraph = dataset_util.generateKNNClusterData(100, 10, 30);
                    return [4, tsne.initializeNeighborsFromKNNGraph(100, 30, knnGraph.distances, knnGraph.indices)];
                case 1:
                    _a.sent();
                    return [4, tsne.iterate()];
                case 2:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(5);
                    return [4, tsne.iterate()];
                case 3:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(4);
                    return [4, tsne.iterate()];
                case 4:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(3);
                    return [4, tsne.iterate()];
                case 5:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(2);
                    return [4, tsne.iterate()];
                case 6:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
                    return [4, tsne.iterate()];
                case 7:
                    _a.sent();
                    expect(tsne.exaggerationAtCurrentIteration).toBeCloseTo(1);
                    tsne.dispose();
                    return [2];
            }
        });
    }); });
});
//# sourceMappingURL=tsne_optimizer_test.js.map