#!/usr/bin/env python3
"""
scripts/convert_to_tflite.py
Convert Keras .keras model to TFLite. Use --quantize none|fp16|int8
If int8, provide --representative_dir with images for calibration.
"""

import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

def representative_gen(image_dir, input_size=(224,224), max_samples=100):
    imgs = list(Path(image_dir).glob("*.*"))
    def gen():
        count = 0
        for p in imgs:
            if count >= max_samples:
                break
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, input_size)
            arr = tf.keras.preprocessing.image.img_to_array(im)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            arr = np.expand_dims(arr, 0).astype(np.float32)
            yield [arr]
            count += 1
    return gen

def convert(keras_path, out_path, quantize='none', rep_dir=None):
    model = tf.keras.models.load_model(str(keras_path), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize == 'none':
        tflite_model = converter.convert()
    elif quantize == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
    elif quantize == 'int8':
        if rep_dir is None:
            raise ValueError("Representative dir required for int8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen(rep_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # set inference input/output types to int8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
    else:
        raise ValueError("quantize must be none|fp16|int8")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[DONE] Wrote {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quantize', default='none', choices=['none','fp16','int8'])
    parser.add_argument('--representative_dir', default=None)
    args = parser.parse_args()
    convert(args.keras, args.output, args.quantize, args.representative_dir)
