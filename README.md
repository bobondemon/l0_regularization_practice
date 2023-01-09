**TODO:**

- [ ] coding the computation cost (FLOPS) and sparsity (model size reduction)
- [ ] report the results that shows the pruning benifits, e.g.: same test accuracy but with lower computation cost
- [ ] analize the pruning parameters, $\ln\alpha$
- [ ] check the version of hydra, pytorch, lightning ...

Apply L0 regularization ([Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312)) on CIFAR10 image classification task with GoogleNet model.

The class of L0 gate layer, `sparsereg.model.basic_l0_blocks.L0Gate`, is modified from the author's [repo](https://github.com/AMLab-Amsterdam/L0_regularization/tree/39a5fe68062c9b8540dba732339c1f5def451f1b).
Also, the CIFAR10 training part including model structure and dataloader are modified from [TUTORIAL 4: INCEPTION, RESNET AND DENSENET](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html). I just refactored into [hydra](https://hydra.cc/docs/intro/) and [lightning](https://pytorch-lightning.readthedocs.io/en/latest/)(my style) format.

# Introduction to L0 Regularization
Please see [xxx]() for detailed understanding about the math under the hood.

We prune the output channels of a convolution layer:

<img src="docs/l0_on_conv_output_channel.png" width=90% height=90%>

Then apply these `L0Gate` for pruning channels in inception block:

<img src="docs/inception_block_with_l0gate.png" width=60% height=60%>

Finally, GoogleNet is then constructed by these *gated* inception blocks.


# Usage

## Main Package Version
```
hydra-core             1.2.0
pytorch-lightning      1.8.4.post0
torch                  1.10.1+cu102
torchaudio             0.10.1+cu102
torchmetrics           0.11.0
torchvision            0.11.2+cu102
```

## How to Train
### **Without** L0 Gating Layer
- Training from scratch
    ```
    python train.py ckpt_path=null resume_training=false without_using_gate=true
    ```
- Resume training from ckpt
    ```
    python train.py ckpt_path=path_of_ckpt/last.ckpt resume_training=true without_using_gate=true
    ```

### **With** L0 Gating Layer
- Training from scratch
    ```
    python train.py ckpt_path=null resume_training=false without_using_gate=false
    ```

### 

## How to Test
```
python test.py ckpt_path=path_of_ckpt/last.ckpt
```

## Monitoring with tensorboard
```
tensorboard --logdir ./outputs/train/tblog/lightning_logs/
```

# Results
xxx
