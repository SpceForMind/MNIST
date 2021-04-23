# MNIST
# Short Example of 4-layers NN 

## Run
Type next command:

```bash
python -m run -h
```

It's show:
```bash
MNIST Trainer/Tester

positional arguments:
  {trainer,tester}
    trainer         MNIST Trainer
    tester          MNIST Tester
```

### Train loop
*If you want to train your network type:*
```bash
python -m run trainer -h

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --learning_rate LEARNING_RATE
  --epochs EPOCHS
  --log_interval LOG_INTERVAL
  --path_to_model PATH_TO_MODEL
  --dir_to_save_model DIR_TO_SAVE_MODEL
```
*Example:*
```bash
python -m run trainer --path_to_model train_results/train_2021_4_22_19_3_59 --dir_to_save_model train_results
```

### Test loop
*If you want to train your network type:*
```bash
python -m run tester -h

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --path_to_model PATH_TO_MODEL
```
*Example:*
```bash
python -m run tester --path_to_model train_results/train_2021_4_22_19_3_59
```

### Recognize single-image with num
*Script-options:*
```bash
python -m run recognize -h

optional arguments:
  -h, --help            show this help message and exit
  --nn_model_path NN_MODEL_PATH (you need run it after learning and saving NN-model)
  --img_path IMG_PATH
```
![](https://raw.githubusercontent.com/SpceForMind/MNIST/main/img/3.png)
*Example:*

```bash
python -m run recognize --nn_model_path train_results/train_2021_4_22_19_3_59 --img_path img/3.jpg
```

### Summary
Mnist NN has accuary ~ 98% if you run train loop with defaults train-settings


Defaults settings defined in the `run/__main__.py`


