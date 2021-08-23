import os
import argparse
import sys
import time

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.summary import FileWriter
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# from synthesizer.tacotron2 import Tacotron2
# from synthesizer.hparams import hparams
# from synthesizer.inference import Synthesizer

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# with tf.device('/device:GPU:7'):

# To use a free GPU, export the following ENV VAR and replace with GPU ID on the CLI
# export CUDA_VISIBLE_DEVICES=<GPU_ID>

# To avoid running into errors like: "No device assignments were active during op <op_name>"
# add allow_soft_place=True, see https://stackoverflow.com/questions/44873273/what-do-the-options-in-configproto-like-allow-soft-placement-and-log-device-plac
tf_config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True, 
    log_device_placement=True
)

last_checkpoint_dir = "Lip2Wav-master/gpu_trained_model"
last_checkpoint_meta = last_checkpoint_dir + "/tacotron_model.ckpt-359000.meta"
last_checkpoint_file = "Lip2Wav-master/gpu_trained_model/checkpoint"
# last_checkpoint = "synthesizer/saved_models/logs-final/taco_pretrained/tacotron_model.ckpt-324000"
# last_checkpoint = "synthesizer/saved_models/logs-lip2wav_1gpu_2021_06_02/taco_pretrained/tacotron_model.ckpt-509000"

""""Ascend's Tacotron ConfigProto()"""
# config = tf.compat.v1.ConfigProto()
# custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
# custom_op.name = "NpuOptimizer"
# custom_op.parameter_map["use_off_line"].b = True
# custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
# config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
# config.allow_soft_placement = True
# config.log_device_placement = True

with tf.Session(config=tf_config) as tf_sess:
     
    # model = Tacotron2(checkpoint_fpath, hparams)
    # specs, alignments = model.my_synthesize(embeddings, texts)

    # Inputs: see hparams.py to get the values (img_size, T, num_gpu, mel_steps)
    # inputs = tf.placeholder(tf.float32, shape=(None, 90, 96, 96, 3), name="inputs"),
    # input_lengths = tf.placeholder(tf.int32, shape=(None,), name="input_lengths"),
    # targets = tf.placeholder(tf.float32, shape=(None, 240, 80), name="mel_targets"),
    # split_infos = tf.placeholder(tf.int32, shape=(1, None), name="split_infos"),
    # speaker_embeddings = tf.placeholder(tf.float32, shape=(None, 256), name="speaker_embeddings")

    # Outputs:
    # text_seq = tf.placeholder(shape=[1, 188], dtype=tf.int32)
    # mel_targets = tf.placeholder(tf.float32, shape=(None, 240, 80), name="mel_targets"),

    """load meta and restore checkpoint"""
    saver = tf.train.import_meta_graph("/home/jeff/Lip2Wav-master/pretrained_model/tacotron_model.ckpt-324000.meta")
    print("'''''''''''''''''CHECKPOINT FILE'''''''''''''''''''''")
    # print(last_checkpoint_file)
    saver.restore(tf_sess, tf.train.latest_checkpoint("/home/jeff/Lip2Wav-master/pretrained_model/checkpoint"))
    # saver.restore(tf_sess, tf.train.latest_checkpoint(last_checkpoint_file))

    g = tf_sess.graph
    g_def = tf_sess.graph.as_graph_def()

    output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=tf_sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=['Tacotron_model/truediv','Tacotron_model/truediv_1', 'Tacotron_model/truediv_2', 'Tacotron_model/truediv_3', 'Tacotron_model/truediv_4']
    )
    tf.io.write_graph(
            output_graph_def,
            logdir='./convert_log',
            name='./tacotron2.pb',
            as_text=False
    )

    """ store the names of each node in the graph as output node """
    # output_node_names = [n.name for n in g_def.node]
    # for n in output_node_names:
    #     print(n)

    """attempt: generate a graph.pb file then use freeze_graph.py to convert
    save graph_def (which includes all variables, as .pb) - think it went ok, try to use freeze_graph.py"""
      

    """attempt: restore from .meta and visualize in tensorboard"""
    """tensorboard --logdir __tb --host=127.0.0.1 """
    # FileWriter("__tb", tf_sess.graph)

  
    """Works: freeze vars -> const and write to .pb"""
    # print('freeze...')
    # frozen_graph = tf.graph_util.convert_variables_to_constants(
    #     tf_sess, 
    #     g_def,
    #     output_node_names)
    # out_graph_path = "GraphDef.pb"
    # with tf.gfile.GFile(out_graph_path, "wb") as f:
    #     f.write(frozen_graph.SerializeToString())     
    # print(f'pb file saved in {out_graph_path}')

""" Validate correctness of .pb - load and print the operations """
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
# print("load graph")
# graph = load_graph('GraphDef.pb')

# ops = graph.get_operations()
# for op in ops:
#     print("\nOp name: ", op.name)
#     print("Op value: ", op.values)
#     print("Op Output: ", op.outputs)



############################################################################################

def freeze_graph_name(input_checkpoint):
    '''
    :param input_checkpoint: 
    '''
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(  
            sess=sess,
            input_graph_def=input_graph_def,# equals:sess.graph_def
            output_node_names=[var.name[:-2] for var in tf.global_variables()]
            )
                 # View all nodes
        for op in graph.get_operations():
            print(op.name, op.values())

# if __name__ == '__main__':
# 	input_checkpoint = 'gpu_trained_model/tacotron_model.ckpt-359000'
# 	freeze_graph_name(input_checkpoint)
