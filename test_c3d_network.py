import tensorflow as tf
from c3d_network import C3D_Network
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader('./models/pretrain/sports1m_finetuning_ucf101.model')
var_to_shape_map = reader.get_variable_to_shape_map()
print(var_to_shape_map['var_name/wc4b'])
print(reader.get_tensor('var_name/wc4b'))

# for key in var_to_shape_map:
#     print(key)
#     #print("tensor_name: ", key)
#     print(reader.get_tensor(key))
