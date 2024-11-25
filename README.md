---
title: Open science statement

---

In the spirit of promoting open science and transparency, we are committed to making the following artifacts available to the research community.

### Code

We will make the source code for the proposed framework and the attacks available. This includes the Python code for implementing the methods described in this paper and the scripts required to run the code. 
The code is now accessible via an **anonymous GitHub repository**: https://anonymous.4open.science/r/W-I-R-309D/README.md
1. Code structure: The code is made of three modules, train module (for training the WCR and WIR models), certify module (for certifying watermark extraction), and attack module (for identity linking, identity forge, and identity extraction attacks). 
  The details of the file's function are described as follows:
  - **Train module:**
    ```
    train/              
        options/  <Train options>
            Hidden/  <Hidden train setting>
                OO_opt.yml  <Clean vanilla setting> 
                Dmi_opt.yml  <WIR setting>
                D_opt.yml  <WCR setting>
            Stega/ <Stegastamptrain setting>
                Dmi_opt.yml  <WIR setting>
                OO_opt.yml  <Clean vanilla setting>
                D_opt.yml  <WCR setting>
        stega_train.py  <Stegastamp training file>
        hidden_train.py  <Hidden training file>
        scripts/  <For saving training scripts>
            hidden/
                Train_mi.sh  <WIR scripts>
                Train.sh  <WCR, WER, Clean vanilla scripts>
            stega/
                Train_mi.sh  <WIR scripts>
                Train.sh  <WCR, WER, Clean vanilla scripts>
        val.py  <Val and test file>
        
  - **Certify:**
    ```
    certify/
        certified_acc.ipynb  <Caculate certified acc file>
        certify_wm.py  <Certify watermark file>
        output/ <For saving results>
        all_datasets.py  <Datasets file>
        WM_smooth.py  <Random smoothing file>
        scripts/
            certify.sh  <Certify scripts>
            
  - **Attack:**
    ```
    attack/
        clustering_function.py  <Functions of cluster>
        ae_data/ <For saving ae (adversarial examples)>
        create_ae.py  <Generate ae >
        extract_attack.ipynb  <Identity extraction attack>
        functions.py  <Functions file>
        forge.ipynb  <Identity forge attack>
        linking.ipynb  <Identity linking attack>
        scripts/
            create_ae.sh  <Gengerate ae scripts>
  - **Others:**
    ```
    models/
        noise_layers/  <noise layers of watermark models>
            resize.py 
            jpeg_compression.py  
            dropout.py  
            noiser.py  
            crop.py  
            cropout.py  
            quantization.py  
            dct_filters.py  
        hidden.py  <Hidden model file>
        hidden_noiser.py  
        deform.py  <Deform wrapper file>
        conv_bn_relu.py  
        stega.py  <Stega model file>
    weights/
        weight/
            CelebA/
                stega/
                    epoch_99_state.pth  <Model weights>
        miweight/
    utils/
        vgg_loss.py  <VGG loss file>
        metrics.py  <Metrics file>
        hidden_options.py  <Hidden options file>
        __init__.py  <Init file>
        Hidden_trainer.py  <Hidden training launcher file>
        log.py  <Log file>
        certifyutils.py  <Certify utils file>
        Stega_trainer.py  <Stega training laucher file>
        yml.py  <Option file>
        helpers.py  <Helpers file>
        stegautils.py  <Stega utils file>
    setting.py <Dataset root dir and resolution setting>
    dataset.py <CustomDataset class>
    requirements.txt <Python packages>
    
2. Create a conda environment:
    ```
    conda create --name WIR --file requirements.txt
    ```

3. Activate conda environment：
    ```
    conda activate WIR
    ```
    
### Datasets
    
We use three publicly available datasets: CelebA-HQ, MSCOCO2017, and ImageNet1K. The specific settings and configurations used for these datasets are detailed in the paper. Additionally, the code for dataset splitting is provided. For example, the CelebA-HQ dataset is divided into a 12:3:1 ratio. These datasets are publicly available and can be accessed through their respective official repositories.
1. Download the datasets from their homepage.
  CelebA-HQ: https://github.com/tkarras/progressive_growing_of_gans
  MS-COCO2017: https://cocodataset.org/#home
  ImageNet-1k: https://www.image-net.org/download.php
2. Unzip the datasets into your datasets folder, then set the path to your dataset in setting.py.
    ```
    ROOT_DIR = "Your datasets folder for saving three datasets"
    ```

### Configuration Information and Scripts
    
To enable the reproducibility of our results, we will provide all configuration information necessary to recreate the experiments. This includes parameters, hyperparameters used in our experiments, as well as details on how to run the proposed framework. In addition, we will make all scripts available in our code.

1. For training the watermark model, you can adjust the parameters and hyperparameters by updating the settings in the options or via the command line in the scripts. 
    ```
    CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset imagenet --batch_size 120 --sigma 0.03 --noise_choice affine \
    --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth
    ```
    The following is a description of some parameters in the configuration file:
    - file: The file path for training settings.
    - sigma: The intensity of noise added during training.
    - lre: The learning rate of the encoder.
    - lrd: The learning rate of the decoder.
    - num: The size of the training set.
    - resume_path: The checkpoint path to resume from.
    

    Then train the WCR model by running in a conda environment:
    ```
    bash scripts/stega/Train.sh
    ```
    
3. For the certified watermark (WCR and WIR), the parameters can be adjusted in the scripts.
    ```
    bash certify/scripts/certify.sh
    ```
    
4. For the three attacks proposed in the paper, you can directly adjust the settings in the provided IPython notebook file, i.e., `linking.ipynb`, `forge.ipynb`, and `extract_attack.ipynb`.

    
### Sharing Limitations

At this time, all the artifacts outlined above are planned for release.
We are committed to ensuring that the open science artifacts are accessible, and we will update the community should there be any changes in the availability of these resources. We aim to foster an environment where our research is transparent, reproducible and can be further developed by the scientific community.
