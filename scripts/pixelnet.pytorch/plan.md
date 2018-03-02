####TODO
1. ~~load pre-trained vgg-16 model and test it on the bsd 500 test set~~
2. ~~implement hypercolumn & **pixel sampling** (pixel sampling is a generic strategy to reduce computation time during training ... so we can combine it with other model, e.g. FCN ...)~~
3. dataset & training script for bsd500
    - ~~need to convert ground truth from matlab to python first~~ **Use the preprocessed HED dataset for now**
    - ~~data augmentation scheme and how to use the ground truth label?~~
    - ~~dataset & loader for the test set -- also on the augmented dataset from HED~~

4. ~~set different learning rates for different part of the model, i.e. small learning rate for the feature network (vgg16) and relatively large learning rate for the classifiers on hypercolumn features.~~

5. **Trimmed VGG** No need to load parameters of the fully connected layers -- customize the torch vgg net.

6. Non-Maximum Suppresion to thinner edges.

####Batch normalization before classifiers atop hypercolumn features
Adding batch normalization before the classifiers atop hypercolumn improves the results visually.

####How to set different learning rate within a model
http://pytorch.org/docs/optim.html

####Pixel Sampling
The naive implementation works, but there is a lot of room to improve in terms of computation/memory efficiency.

reference: [PixelNet: Representation of the pixels, by the pixels, and for the pixels.](http://www.cs.cmu.edu/~aayushb/pixelNet/pixelnet.pdf)

####Hypercolumn features
reference: [Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/abs/1411.5752)
