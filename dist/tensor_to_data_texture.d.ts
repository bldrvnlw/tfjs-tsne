import * as tf from '@tensorflow/tfjs-core';
import { RearrangedData } from './interfaces';
export declare function tensorToDataTexture(tensor: tf.Tensor): Promise<{
    shape: RearrangedData;
    texture: WebGLTexture;
}>;
