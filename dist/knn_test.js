"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var gl_util = require("./gl_util");
var tf_knn = require("./knn");
function iterate(knn, knnTechnique) {
    if (knnTechnique === 'brute force') {
        knn.iterateBruteForce();
    }
    else if (knnTechnique === 'random sampling') {
        knn.iterateRandomSampling();
    }
    else if (knnTechnique === 'knn descent') {
        knn.iterateKNNDescent();
    }
    else {
        throw new Error('unknown knn technique');
    }
}
function knnIntegrityTests(knnTechnique, dataTexture, dataFormat, numPoints, numDimensions, numNeighs) {
    it("kNN increments the iterations\n      (" + knnTechnique + ", #neighs: " + numNeighs + ")", function () {
        var knn = new tf_knn.KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
        iterate(knn, knnTechnique);
        expect(knn.iteration).toBe(1);
        iterate(knn, knnTechnique);
        expect(knn.iteration).toBe(2);
    });
    it("kNN preserves the heap property\n      (" + knnTechnique + ", #neighs: " + numNeighs + ")", function () {
        var knn = new tf_knn.KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        tf.tidy(function () {
            var distancesTensor = knn.distancesTensor();
            expect(checkHeap(distancesTensor, numPoints, numNeighs)).toBe(true);
        });
    });
    it("kNN does not have duplicates\n      (" + knnTechnique + ", #neighs: " + numNeighs + ")", function () {
        var knn = new tf_knn.KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        tf.tidy(function () {
            var indices = knn.indicesTensor();
            expect(checkDuplicates(indices, numPoints, numNeighs)).toBe(true);
        });
    });
    it("kNN does not have invalid neighbors\n      (" + knnTechnique + ", #neighs: " + numNeighs + ")", function () {
        var knn = new tf_knn.KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, numNeighs, false);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        iterate(knn, knnTechnique);
        tf.tidy(function () {
            var indices = knn.indicesTensor();
            expect(checkInvalidNeighbors(indices, numPoints, numNeighs))
                .toBe(true);
        });
    });
}
describe('KNN [line]\n', function () {
    var numDimensions = 12;
    var pointsPerRow = 10;
    var numRows = 100;
    var numPoints = pointsPerRow * numRows;
    var vec = new Uint8Array(pointsPerRow * numDimensions * numRows);
    for (var i = 0; i < numPoints; ++i) {
        for (var d = 0; d < numDimensions; ++d) {
            vec[i * numDimensions + d] = 255. * i / numPoints;
        }
    }
    var backend = tf.ENV.findBackend('webgl');
    var gpgpu = backend.getGPGPUContext();
    var dataTexture = gl_util.createAndConfigureUByteTexture(gpgpu.gl, pointsPerRow * numDimensions / 4, numRows, 4, vec);
    var dataFormat = {
        numPoints: numPoints,
        pointsPerRow: pointsPerRow,
        pixelsPerPoint: numDimensions / 4,
        numRows: numRows,
    };
    it('Checks for too large a neighborhood', function () {
        expect(function () {
            new tf_knn.KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, 129, false);
        }).toThrow();
    });
    it('k must be a multiple of 4', function () {
        expect(function () {
            new tf_knn.KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, 50, false);
        }).toThrow();
    });
    it('kNN initializes iterations to 0', function () {
        var knn = new tf_knn.KNNEstimator(dataTexture, dataFormat, numPoints, numDimensions, 100, false);
        expect(knn.iteration).toBe(0);
    });
    knnIntegrityTests('brute force', dataTexture, dataFormat, numPoints, numDimensions, 100);
    knnIntegrityTests('random sampling', dataTexture, dataFormat, numPoints, numDimensions, 100);
    knnIntegrityTests('knn descent', dataTexture, dataFormat, numPoints, numDimensions, 100);
    knnIntegrityTests('brute force', dataTexture, dataFormat, numPoints, numDimensions, 48);
    knnIntegrityTests('random sampling', dataTexture, dataFormat, numPoints, numDimensions, 48);
    knnIntegrityTests('knn descent', dataTexture, dataFormat, numPoints, numDimensions, 48);
});
function linearTensorAccess(tensor, i) {
    var elemPerRow = tensor.shape[1];
    var col = i % elemPerRow;
    var row = Math.floor(i / elemPerRow);
    return tensor.get(row, col);
}
function checkHeap(distances, numPoints, numNeighs) {
    var eps = 10e-5;
    for (var i = 0; i < numPoints; ++i) {
        var s = i * numNeighs;
        for (var n = 0; n < numNeighs; ++n) {
            var fatherId = s + n;
            var fatherValue = linearTensorAccess(distances, s + n);
            var sonLeftId = s + n * 2 + 1;
            var sonRightId = s + n * 2 + 2;
            if (sonLeftId < numNeighs &&
                fatherValue - linearTensorAccess(distances, sonLeftId) < -eps) {
                distances.print();
                console.log("fatherAbs " + (fatherId - s));
                console.log("fatherId " + fatherId);
                console.log(fatherValue);
                console.log("sonAbs " + (sonLeftId - s));
                console.log("sonId " + sonLeftId);
                console.log(linearTensorAccess(distances, sonLeftId));
                return false;
            }
            if (sonRightId < numNeighs &&
                fatherValue - linearTensorAccess(distances, sonRightId) < -eps) {
                distances.print();
                console.log("fatherAbs " + (fatherId - s));
                console.log("fatherId " + fatherId);
                console.log(fatherValue);
                console.log("sonAbs " + (sonRightId - s));
                console.log("sonId " + sonRightId);
                console.log(linearTensorAccess(distances, sonRightId));
                return false;
            }
        }
    }
    return true;
}
function checkDuplicates(indices, numPoints, numNeighs) {
    var duplicates = 0;
    for (var i = 0; i < numPoints; ++i) {
        var s = i * numNeighs;
        for (var n = 0; n < numNeighs; ++n) {
            for (var n1 = n + 1; n1 < numNeighs; ++n1) {
                var value = linearTensorAccess(indices, s + n);
                var value1 = linearTensorAccess(indices, s + n1);
                if (value === value1 && value1 !== -1) {
                    duplicates++;
                }
            }
        }
    }
    if (duplicates !== 0) {
        console.log("Duplicates:\t " + duplicates);
        console.log("Duplicates per point:\t " + duplicates / numPoints);
        return false;
    }
    return true;
}
function checkInvalidNeighbors(indices, numPoints, numNeighs) {
    var invalid = 0;
    for (var i = 0; i < numPoints; ++i) {
        var s = i * numNeighs;
        for (var n = 0; n < numNeighs; ++n) {
            var value = linearTensorAccess(indices, s + n);
            if (value === -1) {
                invalid++;
            }
        }
    }
    if (invalid !== 0) {
        console.log("Invalid:\t " + invalid);
        console.log("Invalid per point:\t " + invalid / numPoints);
        return false;
    }
    return true;
}
//# sourceMappingURL=knn_test.js.map