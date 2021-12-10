# DePerceiver: Improving Small Object Detection in DeTR

This is the PyTorch Lightning code for DePerceiver. DePerceiver is a model that performs object detection by modifying the implementation of [DeTR](https://github.com/facebookresearch/detr). Although DeTR performs on par with state of the art object detection models like Faster R-CNN, it lags behind in small object detection. This is due to the downsampling in the backbone; when its image features are fed into the transformers, it is difficult to use the low resolution features to detect small objects. DeTR uses a transformer, whose self-attention module is quadratic with respect to image size, which does not allow it to use features at multiple scales.

We propose DePerceiver, which makes use of the Perceiver IO architecture in place of the transformer. Perceiver IO is a transformer-like architecture that has linear complexity, which allows the model to run more efficiently and to use larger feature maps (or multi-scale).

This code is our attempt to improve on object detection baselines with the DePerceiver model. The COCO 2017 dataset needs to be downloaded to train and evaluate the models in this repository. Once that is complete, the model can be trained with `main_naive.py`. The implementation details of the architecture are present under the `deperceiver/` folder.

For a more detailed description of the project see our writeup here : ADD LINK TO PAPER

This code was submitted as a final project by Edward Li and Aditya Kannan for Visual Learning and Recognition (16-824) in Fall 2021.
