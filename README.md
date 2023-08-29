# Neighborhood Contrastive Transformer for Change Captioning
This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["Neighborhood Contrastive Transformer for Change Captioning."](https://ieeexplore.ieee.org/document/10086696), which has appeared as a regular paper in IEEE TMM 2023. 

## We illustrate the training details as follows:

## Installation
1. Clone this repository
2. cd NCT
1. Make virtual environment with Python 3.8 
2. Install requirements (`pip install -r requirements.txt`)
3. Setup COCO caption eval tools ([github](https://github.com/mtanti/coco-caption)) 
4. An 3090 GPU or others.

## Data
1. Download data from here: [google drive link](https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view?usp=sharing)
```
python google_drive.py 1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe clevr_change.tar.gz
tar -xzvf clevr_change.tar.gz
```
Extracting this file will create `data` directory and fill it up with CLEVR-Change dataset.

2. Preprocess data

We are providing the preprocessed data here: [google drive link](https://drive.google.com/file/d/1FA9mYGIoQ_DvprP6rtdEve921UXewSGF/view?usp=sharing).
You can skip the procedures explained below and just download them using the following command:
```
python google_drive.py 1FA9mYGIoQ_DvprP6rtdEve921UXewSGF ./data/clevr_change_features.tar.gz
cd data
tar -xzvf clevr_change_features.tar.gz
```

* Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

* Build vocab and label files using caption annotations:
```
python scripts/preprocess_captions_dep_transformer.py
```

## Training
To train the proposed method, run the following commands:
```
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training
python train_trans_syntax.py --cfg configs/dynamic/transformer_syntax.yaml 
```


## Testing/Inference
To test/run inference on the test dataset, run the following command
```
python test_syntax.py --cfg configs/dynamic/transformer_syntax.yaml  --visualize --snapshot 6000 --gpu 1
```
The command above will take the model snapshot at 6000th iteration and run inference using GPU ID 1, saving visualizations as well.

## Evaluation
* Caption evaluation

To evaluate captions, we need to first reformat the caption annotations into COCO eval tool format (only need to run this once). After setting up the COCO caption eval tools ([github](https://github.com/tylin/coco-caption)), make sure to modify `utils/eval_utils.py` so that the `COCO_PATH` variable points to the COCO eval tool repository. Then, run the following command:
```
python utils/eval_utils.py
```

After the format is ready, run the following command to run evaluation:
```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate.py --results_dir ./experiments/NCT/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```

Once the best model is found on the validation set, you can run inference on test set for that specific model using the command exlpained in the `Testing/Inference` section and then finally evaluate on test set:
```
python evaluate.py --results_dir ./experiments/NCT/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```
The results are saved in `./experiments/R3NET+SSP/test_output/captions/eval_results.txt`

If you find this helps your research, please consider citing:
```
@ARTICLE{tu2023neighborhood,
  author={Tu, Yunbin and Li, Liang and Su, Li and Lu, Ke and Huang, Qingming},
  journal={IEEE Transactions on Multimedia}, 
  title={Neighborhood Contrastive Transformer for Change Captioning}, 
  year={2023},
  pages={1-12},
  doi={10.1109/TMM.2023.3254162}
  }
```

## Contact
My email is tuyunbin1995@foxmail.com

Any discussions and suggestions are welcome!


