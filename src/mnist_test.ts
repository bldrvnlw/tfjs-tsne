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
import * as tf_tsne from './tsne';

// disable tslint for a single line permitting require import of library
//tslint:disable-next-line
const pngReader = require('pngjs').PNG;
jasmine.DEFAULT_TIMEOUT_INTERVAL = 3000000;

const imgSize = 512;
function sleep(time: number) {
  return new Promise(resolve => setTimeout(resolve, time));
}

// assume data is arranged along axis 0
// returns a new shuffled tensor
function shuffleTensorData(tensor: tf.Tensor) {
  const size = tensor.shape[0];
  const seq = tf.util.createShuffledIndices(size);
  //let seq = fisherShuffle([...(new Int32Array(size)).keys()]);
  return tensor.gather(tf.tensor(Array.from(seq), [size], 'int32'));
}

function subsampleTensorImages(tensor: tf.Tensor,
                               oldWidth: number,
                               oldHeight: number,
                               newWidth: number,
                               newHeight:number,
                               numImages: number) {
  const subSet = tensor.slice([0,0], [numImages]).
    as4D(numImages, oldHeight, oldWidth, 1);
  return subSet.resizeBilinear([newHeight, newWidth]).
    reshape([numImages, newWidth*newHeight]);
}

function initCanvas() {
  // create drawing canvas of required dimensions
  const canv = document.createElement('canvas');
  canv.height = imgSize;
  canv.width = imgSize;
  canv.id = 'plotCanv';
  document.body.appendChild(canv);
  return canv.getContext('2d');
}

function plotCoords(numberPoints: number, coordData: any, ctx: any) {
  const imgData = ctx.getImageData(0,0,ctx.canvas.width, ctx.canvas.height);
  const pixArray = new Uint8ClampedArray(imgData.data.buffer);
  // zero the buffer for the cumulative plot (black
  for (let i=0; i<pixArray.byteLength; i++) {
    (i % 4) === 3 ? pixArray[i] = 255: pixArray[i] = 0;
  }
  //non-linear accumulate to a maximum of 255 by adding 1/2 the diff from 255
  for (let i=0; i<numberPoints*2; i+=2) {
    const xcoord = Math.round(coordData[i] * (imgSize-1));
    const ycoord = Math.round(coordData[i+1] * (imgSize-1));
    // ImageData is RGBA
    const offset = 4 * (ycoord * ctx.canvas.width + xcoord);
    pixArray[offset] = pixArray[offset] +
      Math.min(100, 255 - pixArray[offset]);
    pixArray[offset + 1] = pixArray[offset];
    pixArray[offset + 2] = pixArray[offset];
    pixArray[offset + 3] = 255;
  }
  //rewrite canvas with new ImageData
  ctx.putImageData(imgData, 0, 0);
}

async function loadMnist() {
  const resp = await fetch('images/mnist_images.png');
  const imgArray = await resp.arrayBuffer();
  const reader = new pngReader();
  return new Promise ((resolve) => {
    reader.parse(imgArray, (err: any, png: any) => {
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

describe('TSNE loading MNIST', () => {

  it('works', async (done) => {

    let dataSet = await loadMnist();
    const allMnistTensor = tf.tensor(dataSet as Float32Array).
      reshape([65000, 784]);
    dataSet = new Float32Array(0);
    console.log(`Input freed: ${dataSet.toString()}`);
    const shuffledValue = shuffleTensorData(allMnistTensor);
    tf.dispose(allMnistTensor);
    // subset and downsample the images
    // Chrome fails >= 19000 , 10, 10,
    // Firefox works up to 60000
    const numberData = 18000;
    const subTensor = subsampleTensorImages(shuffledValue,
      28,
      28,
      10,
      10,
      numberData);

    console.log(`calculati0ng on: ${subTensor.shape}`);

    const testOpt = tf_tsne.tsne(subTensor, {
      perplexity : 30,
      verbose : true,
      knnMode : 'auto',
    });

    const knnIterations = testOpt.knnIterations();
    for(let i=0; i<knnIterations;
        i+=(50 < knnIterations-i)?knnIterations-i:50) {
      await testOpt.iterateKnn((50 < knnIterations-i)?knnIterations-i:50);
      await sleep(1);
    }

    // get the image data and access the data buffer to overwrite
    const plotCtx = initCanvas();
    await sleep(1);

    for(let i=0; i<1000; i+=1) {
      await testOpt.iterate(1);
      const coordData = await testOpt.coordinates().data();
      plotCoords(numberData, coordData, plotCtx);
      await sleep(1);
    }
    await sleep(5000);
    console.log(`Tsne done`);
    tf.dispose(subTensor);
    done();
  });
});