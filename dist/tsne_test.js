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
var tf_tsne = require("./tsne");
function generateData(numPoints, numDimensions) {
    if (numPoints === void 0) { numPoints = 300; }
    if (numDimensions === void 0) { numDimensions = 10; }
    var data = tf.tidy(function () {
        return tf.linspace(0, 1, numPoints * numDimensions)
            .reshape([numPoints, numDimensions])
            .add(tf.randomUniform([numPoints, numDimensions]));
    });
    return data;
}
describe('TSNE class', function () {
    it('throws an error if the perplexity is too high', function () {
        var data = generateData();
        expect(function () {
            tf_tsne.tsne(data, {
                perplexity: 100,
                verbose: false,
                knnMode: 'auto',
            });
        }).toThrow();
        data.dispose();
    });
});
describe('TSNE class', function () {
    it('throws an error if the perplexity is too high on this system ', function () {
        var data = generateData();
        var maximumPerplexity = tf_tsne.maximumPerplexity();
        expect(function () {
            tf_tsne.tsne(data, {
                perplexity: maximumPerplexity + 1,
                verbose: false,
                knnMode: 'auto',
            });
        }).toThrow();
        data.dispose();
    });
});
describe('TSNE class', function () {
    it('does not throw an error if the perplexity is set to the maximum value', function () {
        var data = generateData();
        var maximumPerplexity = tf_tsne.maximumPerplexity();
        expect(function () {
            tf_tsne.tsne(data, {
                perplexity: maximumPerplexity,
                verbose: false,
                knnMode: 'auto',
            });
        }).not.toThrow();
        data.dispose();
    });
});
describe('TSNE class', function () {
    it('iterateKnn and iterate also work when the number of ' +
        'dimensions is larger than the number of points', function () { return __awaiter(_this, void 0, void 0, function () {
        var data, testOpt, e_1, e_2, coords;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    data = generateData(100, 20000);
                    testOpt = tf_tsne.tsne(data, {
                        perplexity: 15,
                        verbose: false,
                        knnMode: 'auto',
                    });
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, testOpt.iterateKnn(10)];
                case 2:
                    _a.sent();
                    return [3, 4];
                case 3:
                    e_1 = _a.sent();
                    fail('iterateKnn threw exception: ${e}');
                    return [3, 4];
                case 4:
                    _a.trys.push([4, 6, , 7]);
                    return [4, testOpt.iterate(10)];
                case 5:
                    _a.sent();
                    return [3, 7];
                case 6:
                    e_2 = _a.sent();
                    fail('iterate threw exception: ${e}');
                    return [3, 7];
                case 7: return [4, testOpt.coordinates()];
                case 8:
                    coords = _a.sent();
                    expect(coords.shape[0]).toBe(100);
                    expect(coords.shape[1]).toBe(2);
                    data.dispose();
                    return [2];
            }
        });
    }); });
});
//# sourceMappingURL=tsne_test.js.map