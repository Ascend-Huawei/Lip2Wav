Deliverable results for GPU training: [README_GPU](https://rnd-gitlab-ca-g.huawei.com/hispark/model_training_hq/-/blob/master/Lip2Wav/1_gpu_training/README_GPU.md), supporting document in folder [1_gpu_training](1_gpu_training)

**[Google Drive link for checkpoint files and .pb files](https://drive.google.com/drive/folders/13dnqFc3WtEFE9dCbvVNQd4q5sDsbFmmF?usp=sharing)**

Deliverable results for NPU training: [README_NPU](https://rnd-gitlab-ca-g.huawei.com/hispark/model_training_hq/-/blob/master/Lip2Wav/2_npu_training/README_GPU.md), supporting document in folder [2_npu_training](2_npu_training)

**NPU Training:**

**Code changes after using conversion tool:**

| Observed Issue  | Code Change | 
| --------  | ------------------- |
| Datafeeder (feeder.py): NPU does not support tf.FIFOQueue for data buffering and queueing. | Changed to buitlin python queue implementation class:Queue for building training and evaluation queues. During training (in synthesizer/train.py) made code changes to load feed_dict from the python queue before running sess.run  | 
| data      | Some long data here | more data             | 

FIFOQueue not supported

Dynamic decode 

pythonic way of slicing in tacohelper and tester

For model conv : py_func op is not supported



