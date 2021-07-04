# GPU Training: Lip2Wav
[[Lip2Wav Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pdf)] | [[Lip2Wav GitHub Repo](https://github.com/Rudrabha/Lip2Wav)]

## Table of Contents
### GPU Training
**[Dataset Download](#dataset-download)**<br>
**[Training Script & Command](#gpu-training-script-and-command)**<br>
**[requirements.txt](#requirement.txt)**<br>
**[loss+perf_gpu.txt](#gpu-loss-and-performance)**<br>
**[checkpont_gpu](#gpu-checkpoint)**<br>
**[Evaluation Metric](#evaluation-metric)**
**[Frozen Graph](#frozen-graph)**

### Checkpoint, .pb, .pbtxt Files 
**[Google Drive link for checkpoint files and .pb files](https://drive.google.com/drive/folders/13dnqFc3WtEFE9dCbvVNQd4q5sDsbFmmF?usp=sharing)**

<hr>

### Dataset Download
Lip2Wav dataset are encrypted YouTube videos for five lecture speakers (chess, chem, deep learning, ethical hacking, hardware). To download a speaker dataset, run `sh download_speaker.sh Dataset/<speaker>` (replace <speaker> with one of five speaker option). Then run `preprocess.py` to preprocess the raw dataset. We provide a preprocessed chess dataset from our experiment, it is stored on the GPU / NPU servers. Below are the locations for the preprocessed chess dataset:

 - GPU server (103): 

    original: `/fastdata/jeff/Dataset/chess`

    softlink: `/home/jeff/Lip2wav-master/Dataset/chess/processed`


 - Ascend NPU server (jiayansuo): 

    original: `/data/lip2wav/chess`

    softlnk: `/home/jiayansuo/Lip2wav_train/Lip2Wav-master_npu_20210602003231/Dataset/chess`

### GPU training script and command
On the 103 GPU server, login as user `jeff` and activate virtual environment `conda activate lip2wav`. The training script (`train.py`) and bash script (`run_1p.sh`) is in project directory: `/home/jeff/Lip2Wav-master`. To begin training, run: 

    bash run_1p.sh

### requirement.txt
Located in: `/home/jeff/Lip2Wav-master` on 103 GPU server. For more details, see the [Installation](#installation) section


### GPU Loss and Performance
Loss and performance are saved in `nohup.out` in: `/home/jeff/Lip2Wav-master`

### GPU Checkpoint
 Pretrained Model (324,000 steps, from paper. Located in `/home/jeff/Lip2Wav-master/pretrained_model/checkpoint`

 GPU Trained (359,000 steps, trained on server). Located in `/home/jeff/Lip2Wav-master/gpu_trained_model/checkpoint`

 ## Installation
 1. Create conda venv and install the appropriate CUDA Toolkit and CuDNN for this repo. (Install `numpy` and `cython` independently to avoid running into conflict when installing from requirements.txt)

        conda create -n lip2wav python=3.7
        conda activate lip2wav
        conda install cudatoolkit=10.0.130 -y
        conda install cudnn=7.6.0=cuda10.0_0
        conda install numpy==1.16.4 cython
        pip install -r requirements.txt

2. Download the dataset using the provided script and run preprocessing. Batch size and number of GPUs can be configured during preprocessing.

       sh download_speaker.sh Dataset/chess
       
       python preprocess.py --speaker_root Dataset/chess --speaker chess --batch_size 128 --ngpu 3

3. Generate ground truth test split

    For pretrained model:

       python complete_test_generate.py -d Dataset/chess -r Dataset/chess/test_results --preset synthesizer/presets/chess.json --checkpoint /home/jeff/Lip2Wav-master/pretrained_model/tacotron_model.ckpt-324000

    For trained model:

       python complete_test_generate.py -d Dataset/chess -r Dataset/chess/test_results_trained_mod --preset synthesizer/presets/chess.json --checkpoint /home/jeff/Lip2Wav-master/synthesizer/saved_models/final_model/tacotron_model.ckpt-360000

4. Training on the preprocessed dataset

       python train.py <name_of_run> --data_root Dataset/chess/ --preset synthesizer/presets/chess.json

    or run the training script

        bash run_1p.sh

5. Compute evaluation metrics: `PESQ`, `ESTOI`, `STOI` on pretrained and trained model
    
    Compute score for pretrained model:

       python score.py -r Dataset/chess/test_results
    
    Compute score for trained model:

       python score.py -r Dataset/chess/test_results_trained_mod/


### Evaluation Metrics
The table below shows the evaluation metrics: `PESQ`, `ESTOI`, `STOI` from 3 experiments: paper, pretrained model, trained model. 

| Experiment        | Mean PESQ | Mean STOI | Mean ESTOI |  Steps  |
|:-----------------:|:---------:|:---------:|:----------:|:-------:|
| Lip2Wav Paper     | **1.400** | **0.418** | **0.290**  | ------
| Pretrained Model  |   1.387   |   0.350   |   0.240    | 324,000 |
| Trained Model     |   1.385   |   0.409   |   0.285    | 360,000 |

### Frozen Graph
Obtain the frozen graph `.pb` and `.pbtxt` by running:

      python freeze_graph.py --meta_file <checkpoint_meta_file_path> --ckpt_dir <checkpoint_directory> --output_pb <output_file_name> --save_pb <true_if_saving_pb> --save_pbtxt <true_if_saving_pbtxt>

By default, `freeze_graph.py` saves a .pb and .pbtxt file. Set `--save_pb False` or `--save_pbtxt False` to disable either one.


