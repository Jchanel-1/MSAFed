# MSAFed
The complete code will be released upon our paper acceptance.

## Datasets
Fed-polyp: You can download Fed-polyp from [fed-polyp](https://drive.google.com/file/d/1w4kuvHaWChP9-OicA9Sxpd8t6v9bGW4G/view?usp=drive_link).

## Pretrain
Train a generalized global model across clients. The learned model will serve as the initialization for subsequent stages.

```
bash pretrain_inside.sh
```

## Inside Adaptation
Fine-tune the pretrained global model on each client using adaptive layer-wise learning rates. This stage outputs the adapted model parameters, multi-client prototypes and the adaptive learning rates.

```
bash ada_inside.sh
```

## Test Time Adaptation
Adapt the global model to outside clients using the prototypes and learning rates obtained from previous stages during test time.

```
bash tta_outside.sh
```
