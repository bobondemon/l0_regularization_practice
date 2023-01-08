**TODO:**

- [ ] coding the computation cost
- [ ] report the results that shows the purning benifits, e.g.: same test accuracy but with lower computation cost
- [ ] analize the purning parameters, $\ln\alpha$

Apply L0 regularization ([Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312)) on CIFAR10 image classification task with GoogleNet model.

The class of L0 gate layer, `sparsereg.model.basic_l0_blocks.L0Gate`, is modified from the author's [repo](https://github.com/AMLab-Amsterdam/L0_regularization/tree/39a5fe68062c9b8540dba732339c1f5def451f1b).
Also, the CIFAR10 training part including model structure and dataloader are modified from [TUTORIAL 4: INCEPTION, RESNET AND DENSENET](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html). I just refactored into [hydra](https://hydra.cc/docs/intro/) and [lightning](https://pytorch-lightning.readthedocs.io/en/latest/)(my style) format.

# Introduction to L0 Regularization
Please see [xxx]() for detailed understanding about the math under the hood.

We purne the output channels of a convolution layer:

<img src="docs/l0_on_conv_output_channel.png" width=60% height=60%>

Then apply these `L0Gate` for purning channels in inception block:

<img src="docs/inception_block_with_l0gate.png" width=40% height=40%>

Finally, GoogleNet is then constructed by these *gated* inception blocks.


# Usage

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