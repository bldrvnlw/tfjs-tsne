"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dataset_util = require("./dataset_util");
describe('dataset_util', function () {
    it('generates a properly shaped kNN for the synthetic cluster', function () {
        var knnGraph = dataset_util.generateKNNClusterData(1000, 10, 100);
        expect(knnGraph.indices.length).toBe(1000 * 100);
        expect(knnGraph.distances.length).toBe(1000 * 100);
    });
    it('generates a properly shaped kNN for the synthetic line', function () {
        var knnGraph = dataset_util.generateKNNLineData(1000, 100);
        expect(knnGraph.indices.length).toBe(1000 * 100);
        expect(knnGraph.distances.length).toBe(1000 * 100);
    });
});
//# sourceMappingURL=dataset_test.js.map