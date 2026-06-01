import tensorflow as tf

print("TensorFlow OK :", tf.__version__)
print("GPU disponible :", tf.config.list_physical_devices('GPU'))
