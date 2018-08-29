/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import * as tf_tsne from '../../src/index';

const pngReader = require('pngjs').PNG;

const imgSize = 512;
const plotDigitIndex = new Int32Array(imgSize*imgSize);
plotDigitIndex.fill(-1);
function sleep(time) {
  return new Promise(resolve => setTimeout(resolve, time));
}

const plotIndex = new Uint16Array(imgSize*imgSize);
// Colors for digits 0-9
const c0 = 0xFF0000;
const c1 = 0xFF9900;
const c2 = 0xCCFF00;
const c3 = 0x33FF00;
const c4 = 0x00FF66;
const c5 = 0x00FFFF;
const c6 = 0x0066FF;
const c7 = 0x3300FF;
const c8 = 0xCC00FF;
const c9 = 0xFF0099;
const colArray = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9];

// 4 levels of color density
function quantizeColors(cols) {
  const quantFac = 4;
  const quantCols = Array(10);
  cols.forEach((val, i) => {
    const rd = Math.floor((val >> 16)/quantFac);
    const gr = Math.floor(((val & 0x00FF00) >> 8)/quantFac);
    const bl = Math.floor((val & 0x0000FF)/quantFac);
    const col = (rd << 16) + (gr << 8) + bl;
    quantCols[i]  = col;
  });
  return quantCols;
}

const quantCols = quantizeColors(colArray);

// Reduce the MNIST images to newWidth newHeight
// and take the first numImages
function subsampleTensorImages(tensor,
                               oldWidth,
                               oldHeight,
                               newWidth,
                               newHeight,
                               numImages) {
  const subSet = tensor.slice([0,0], [numImages]).
    as4D(numImages, oldHeight, oldWidth, 1);
  return subSet.resizeBilinear([newHeight, newWidth]).
    reshape([numImages, newWidth*newHeight]);
}

function initCanvas() {
  const plotCanv = document.getElementById('plotCanv');
  const digitCanv = document.getElementById('digitCanv');
  // create drawing canvas of required dimensions
  const plotCanvCtx = plotCanv.getContext('2d');
  const digitCanvCtx = digitCanv.getContext('2d');
  blankCanvas(plotCanvCtx);
  blankCanvas(digitCanvCtx);
  return {plotCanvCtx: plotCanvCtx, digitCanvCtx: digitCanvCtx};
}

/**
 * Set a canvas context to black and return the associated
 * imageData and underlying data buffer for further manipulation.
 * @param ctx
 * @returns {{imgData: ImageData, pixArray: Uint8ClampedArray}}
 */
function blankCanvas(ctx) {
  const imgData = ctx.getImageData(0,0,ctx.canvas.width, ctx.canvas.height);
  const pixArray = new Uint8ClampedArray(imgData.data.buffer);
  // zero the buffer for the cumulative plot (black
  const fillArray = new Uint32Array(imgData.data.buffer);
  fillArray.fill(0xFF000000); //little endian
  ctx.putImageData(imgData, 0, 0);
  return {imgData: imgData, pixArray: pixArray};
}

/**
 * Cumulatively plot points from coordinates into an image context.
 * Points are colored according to labels.
 *
 * @param numberPoints
 * @param coordData
 * @param labelData
 * @param ctx
 */
function plotCoords(numberPoints,
                    coordData,
                    labelData,
                    ctx) {
  const dataPix = blankCanvas(ctx);
  const imgData = dataPix.imgData;
  const pixArray = dataPix.pixArray;
  plotDigitIndex.fill(-1);
  // zero the buffer for the cumulative plot (black
  for (let i=0; i<pixArray.byteLength; i++) {
    (i % 4) === 3 ? pixArray[i] = 255: pixArray[i] = 0;
  }

  for (let i=0; i<numberPoints*2; i+=2) {
    const xcoord = Math.round(coordData[i] * (imgSize-1));
    const ycoord = Math.round(coordData[i+1] * (imgSize-1));
    // ImageData is RGBA
    const digitIndex = ycoord * ctx.canvas.width + xcoord;
    plotDigitIndex[digitIndex] = i/2;
    const offset = 4 * digitIndex;
    const label = labelData[i/2];
    // Colors are accumulated into the initially black plot pixels
    // for a pseudo density effect.
    const col = quantCols[label];
    pixArray[offset] = pixArray[offset] + (col >> 16);
    pixArray[offset + 1] = pixArray[offset + 1] + ((col & 0x00FF00) >> 8);
    pixArray[offset + 2] = pixArray[offset + 2] + (col & 0x0000FF);
    pixArray[offset + 3] = 255;

  }
  //rewrite canvas with new ImageData
  ctx.putImageData(imgData, 0, 0);
}

/**
 * MNIST labels are stored as 65000x10 onehot encoding
 * convert this to label number
 * @param labels
 * @returns {Uint8Array}
 */
function oneHotToIndex(labels) {
  const res = new Uint8Array(labels.length/10);
  for(let i =0; i<labels.length; i+=10) {
    for (let j = 0; j < 10; j++) {
      if (labels[i+j] === 1) {
        res[i/10] = j;
        break;
      }
    }
  }
  return res;
}

/**
 * Get a promise that loads the MNIST data.
 * @returns {Promise<*>}
 */
async function loadMnist() {
  const resp = await fetch('../../images/mnist_images.png');
  const imgArray = await resp.arrayBuffer();
  const reader = new pngReader();
  return new Promise ((resolve) => {
    reader.parse(imgArray, (err, png) => {
      // parsed PNG is Uint8 RGBA with range 0-255
      // - convert to RGBA Float32 range 0-1
      const pixels = new Float32Array(png.data.length/4);
      for (let i = 0; i < pixels.length; i++) {
        pixels[i] = png.data[i*4]/255.0;
      }
      resolve(pixels);
    });
  });
}

/**
 * Get a promist that loads the MNIST label data
 * @returns {Promise<ArrayBuffer>}
 */
async function loadMnistLabels() {
  const resp = await fetch('../../images/mnist_labels_uint8.bin');
  return resp.arrayBuffer();
}

/**
 * A global to hold the MNIST data
 */
let dataSet;
/**
 * A global to hold the MNIST label data
 */
let labelSet;

let cancel = false;
/**
 * Run tf-tsne on the MNIST and plot the data points
 * in a simple interactive canvas.
 * @returns {Promise<void>}
 */
async function runTsne(plotCtx) {
  cancel = false;
  // The MNIST set is preshuffled
  const allMnistTensor = tf.tensor(dataSet).
    reshape([65000, 784]);
  // subset and downsample the images
  const numberData = 65000;
  const subTensor = subsampleTensorImages(allMnistTensor,
    28,
    28,
    28,
    28,
    numberData);

  console.log(`calculating on: ${subTensor.shape}`);

  const tsneOpt = tf_tsne.tsne(subTensor, {
    perplexity : 30,
    verbose : true,
    knnMode : 'auto',
  });

  const maxKnnIters = document.getElementById('kNNSlider').value;
  const knnIterations = Math.min(tsneOpt.knnIterations(), maxKnnIters);
  const knnIterElement = document.getElementById('knnIterCount');
  for(let i=0; i<knnIterations; i++) {
    await tsneOpt.iterateKnn(1);
    knnIterElement.innerHTML = 'knn iteration: ' + (i + 1);
    if (cancel) {
      cancel = false;
      return;
    }
    await sleep(1);
  }

  const tsneIterElement = document.getElementById('tsneIterCount');
  // get the image data and access the data buffer to overwrite
  for(let i=0; i<1000; i+=1) {
    await tsneOpt.iterate(1);
    const coordData = await tsneOpt.coordinates().data();
    plotCoords(numberData, coordData, labelSet, plotCtx);
    tsneIterElement.innerHTML = 'tsne iteration: ' + (i + 1);
    // allow time for display
    if (cancel) {
      cancel = false;
      return;
    }
    await sleep(1);
  }
  console.log(`Tsne done`);
  tf.dispose(subTensor);
}

/**
 * Plot the digit on the canvas
 *
 * @param digitCtx
 * @param digitData
 */
async function digitOnCanvas(digitCtx, digitData) {
  const height = digitCtx.canvas.height;
  const width = digitCtx.canvas.height;
  const dataPix = blankCanvas(digitCtx);
  const imgData = dataPix.imgData;
  const pixArray = dataPix.pixArray;
  // put the digit data in a tensor and resize it
  // to match the canvas
  const imgTensor = tf.tensor4d(digitData, [1, 28, 28, 1]);

  const resizedTensor = imgTensor.resizeNearestNeighbor([height, width]);
  const resizedArray = await resizedTensor.data();
  resizedArray.forEach((val, idx) => {
    const pixOffset = 4 * idx;
    const pixVal = 255 * val;
    pixArray[pixOffset] = pixVal;
    pixArray[pixOffset + 1] = pixVal;
    pixArray[pixOffset + 2] = pixVal;
  });
  digitCtx.putImageData(imgData, 0, 0);
}
/**
 * Handle the mousemove event to explore the points in the
 * plot canvas.
 * @param plotCanv
 * @param e
 */
function plotExplore(plotCtx, digitCtx, e) {
  const x  = e.clientX - plotCtx.canvas.offsetLeft;
  const y  = e.clientY - plotCtx.canvas.offsetTop;
  const digitIndex = plotDigitIndex[y * plotCanv.width + x];
  if (digitIndex >= 1) {
    console.log(`digit idx: ${digitIndex}, label: ${labelSet[digitIndex]}`);
    const digitData = dataSet.slice(digitIndex*784, (digitIndex+1)*784);
    digitOnCanvas(digitCtx, digitData);
  }
}

function restart(plotCtx) {
  cancel = true;
  setTimeout(async ()=> {
    initCanvas();
    await runTsne(plotCtx)
  }, 1000)

}

function stop() {
  cancel = true;
}

window.onload = async function() {
  const contexts = initCanvas();
  const plotCtx = contexts.plotCanvCtx;
  const digitCtx = contexts.digitCanvCtx;

  dataSet = await loadMnist();
  const labelOneHot = new Uint8Array(await loadMnistLabels());
  labelSet = oneHotToIndex(labelOneHot);

  document.getElementById('kNNSlider').oninput = () => {
    document.getElementById('sliderVal').innerHTML = 'max kNN iterations: ' + document.getElementById('kNNSlider').value;
  }
  document.getElementById('plotCanv').addEventListener('mousemove', plotExplore.bind(null, plotCtx, digitCtx));
  document.getElementById('restartButton').addEventListener('click', restart.bind(null, plotCtx));
  document.getElementById('stopButton').addEventListener('click', stop);
  await runTsne(plotCtx);
}
