from npu_bridge.npu_init import *

from synthesizer.feeder_modified import Feeder
# from synthesizer.hparams import 

import tensorflow as tf
 

def prepare_run(data_root='Dataset/chess/'):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_log_level)
#     run_name = args.name
#     log_dir = os.path.join(args.models_dir, "logs-{}".format(run_name))
#     os.makedirs(log_dir, exist_ok=True)
    all_images = get_image_list('npu_train', args.data_root) #'train' 
    #print ("train images: ", args.data_root, all_images)
    all_test_images = get_image_list('npu_val', args.data_root)
    #print ("test images : ", all_test_images)

    hparams.add_hparam('all_images', all_images)
    hparams.add_hparam('all_test_images', all_test_images)

    print('Training on {} hours'.format(len(all_images) / (3600. * hparams.fps)))
    print('Validating on {} hours'.format(len(all_test_images) / (3600. * hparams.fps)))

    return hparams



# def main():
#     batch_size = 1

#     coord = tf.train.Coordinator()
#     with tf.name_scope('create_inputs'):
#         reader = DataGenerator(coord)
#         print ("reader done!")
#         input_batch = reader.dequeue(batch_size)
#         print ("input batch ", input_batch)
#     config = tf.ConfigProto()
#     custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
#     custom_op.name = "NpuOptimizer"
#     config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
#     config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关
#     config=npu_config_proto(config_proto=config)
#     sess = tf.Session(config=config)
#     init = tf.global_variables_initializer()
#     sess.run(init)
    
#     print ("inside main")
#     if True:
#         print ("T3sxrue")
#         sess.run(reader.enqueue, feed_dict={reader.sample_placeholder: np.random.uniform(size=[5,5])})
#         print ("after sess")
#         a = sess.run(input_batch)
#         print('a value : ', a)
#         return


#     net,net_in = define_net(input_batch)   
#     print ("net and net in : ", net, net_in)
#     threads = reader.start_threads(sess)
#     queue_size = reader.queue_size
#     for step in range(10000):
#         print('size queue =', queue_size.eval(session=sess))
#         print(sess.run(net))
#         break
#         # Make this thread slow. You can comment this line. If you do so, you will dequeue
#         # faster than you enqueue, so expect the queue not to reach its maximum (32 by default)
#         time.sleep(1)

#     coord.request_stop()
#     print("stop requested.")
#     for thread in threads:
#         thread.join()
        
    
if __name__ == "__main__":
    
    hparams = prepare_run(args)
    
    coord = tf.train.Coordinator()
        with tf.variable_scope("datafeeder") as scope:
            feeder = Feeder(coord, hparams)
#             print ("feeder contents : ", feeder)
