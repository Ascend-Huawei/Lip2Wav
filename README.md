Deliverable results for GPU training: [README_GPU](https://rnd-gitlab-ca-g.huawei.com/hispark/model_training_hq/-/blob/master/Lip2Wav/1_gpu_training/README_GPU.md), supporting document in folder [1_gpu_training](1_gpu_training)

**[Google Drive link for checkpoint files and .pb files](https://drive.google.com/drive/folders/13dnqFc3WtEFE9dCbvVNQd4q5sDsbFmmF?usp=sharing)**

Deliverable results for NPU training and supporting document in folder [2_npu_training](2_npu_training)

## NPU Training :

### Dataset Download:

To download a speaker dataset, run sh download_speaker.sh Dataset/<speaker> (replace  with one of five speaker option). Then run preprocess.py to preprocess the raw dataset. We provide a preprocessed chess dataset from our experiment, it is stored on the GPU / NPU servers. Below is the location for the NPU preprocessed chess dataset:

 - Ascend NPU server (jiayansuo): 

    original: `/data/lip2wav/chess`

    softlnk: `/home/jiayansuo/Lip2wav_train/Lip2Wav-master_npu_20210602003231/Dataset/chess`

  
### NPU training command
  
```
cd scripts
bash run_npu_1p.sh
```

### Code changes after using conversion tool:

| Observed Issues  | Code Changes | 
| --------  | ------------------- |
| *Datafeeder (feeder.py):* <br/> NPU does not support tf.FIFOQueue for data buffering and queueing. | Changed to buitlin python queue implementation class::Queue for building training and evaluation queues. During training (in synthesizer/train.py) made code changes to load feed_dict based on queue size. If not empty, load the feed_dict with actual queue values else load defined placeholders before passing to sess.run  | 
| *Dynamic decode (tacotron.py):* <br/> We observed that the tf.dynamic_deocde wrapper is not supported by NPU. Dynamic decode wrapper takes CustomDecoder as input and performs dynamic decoding at each step  | We found tf.while_loop as the replacement for tf.dynamic_decode and was supported by NPU. Implemented tf.while_loop consisting of condition and body functions. The body function outputs the next step, next inputs, outputs, final output and next state. Frames prediction and final decoder state were obtained as outputs from tf.while_loop  | 
| *Dynamic decode (dynamic_decode_test.py):* <br/> Tf.while_loop successfully worked on NPU but degraded the training performance (~higher steps/sec)  | We modified the tf.dynamic_decode souce code. Replaced the source code logic with our tf.while_loop implementation as stated above. Tested this approach on NPU and drastically improved the training performance  |
| *Tensor Slicing in TacoTestHelper (helpers.py):* <br/> NPU does not support pythonic way of slicing tensors. For example: next_inputs = outputs[:, -self._output_dim:] | Replaced next_inputs = outputs[:, -self._output_dim:] as next_inputs = tf.slice(outputs, [0, tf.shape(outputs)[1] -self._output_dim], [self._hparams.tacotron_batch_size, self._output_dim])| 
| *Unsupported tf.py_func operator during Model conversion (change_graph.py):* <br/> tf.py_func takes inputs, split_func function and output float type as inputs and wraps it in the tf graph | Removed py_func operator and logic from the model. Used split_func to get correct tensor dimensions for the placeholders and then used them as input to the model | 



