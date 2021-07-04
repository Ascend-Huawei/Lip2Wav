import tensorflow as tf
from tensorflow.summary import FileWriter
import argparse

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# with tf.device('/device:GPU:7'):

# To use a free GPU, export the following ENV VAR and replace with GPU ID on the CLI
# export CUDA_VISIBLE_DEVICES=<GPU_ID>

# To avoid running into errors like: "No device assignments were active during op <op_name>"
# add allow_soft_place=True, see https://stackoverflow.com/questions/44873273/what-do-the-options-in-configproto-like-allow-soft-placement-and-log-device-plac

def freeze_graph(meta_file_path, ckpt_dir_path, output_pb_name, save_pb, save_pbtxt):
    tf_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=True
    )
    with tf.Session(config=tf_config) as tf_sess:
        saver = tf.train.import_meta_graph(meta_file_path)
        saver.restore(tf_sess, tf.train.latest_checkpoint(ckpt_dir_path))
        model_file = meta_file_path.split(".")[-2]
        print("restored;{}".format(model_file))

        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess=tf_sess,
                input_graph_def=tf.get_default_graph().as_graph_def(),
                output_node_names=['Tacotron_model/truediv','Tacotron_model/truediv_1', 'Tacotron_model/truediv_2', 'Tacotron_model/truediv_3', 'Tacotron_model/truediv_4']
        )

        if save_pb:
            print("Saving .pb...")
            out_graph_path = output_pb_name
            with tf.gfile.GFile(out_graph_path, "wb") as f:
                f.write(output_graph_def.SerializeToString())     
            print(f'pb file {out_graph_path} saved')

        if save_pbtxt:
            print("Saving .pbtxt...")
            tf.train.write_graph(output_graph_def, ".", 'tacotron2.pbtxt', as_text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", type=str, default="/home/jeff/Lip2Wav-master/gpu_trained_model/tacotron_model.ckpt-359000.meta", 
      help="checkpoint .meta file path")
    parser.add_argument("--ckpt_dir", type=str, default="/home/jeff/Lip2Wav-master/gpu_trained_model",
      help="checkpoint directory")
    parser.add_argument("--output_pb", type=str, default="tacotron2.pb", help="Output protobuf filename")  
    parser.add_argument("--save_pb", type=bool, default=True, help="True if saving to .pb")
    parser.add_argument("--save_pbtxt", type=bool, default=True, help="True if saving to .pbtxt")
    args = parser.parse_args()

    freeze_graph(args.meta_file, args.ckpt_dir, args.output_pb, args.save_pb, args.save_pbtxt)