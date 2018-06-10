## Project: Deep Learning

---


# Required Steps for a Passing Submission:
1. Clone the project repo.
2. Implement a segmentation network in model_training.ipynb.
3. Optimize network and hyper-parameters.
4. Train the network.
5. Continue to experiment with the training data and network until the accuracy is greater than or equal to 40%.
6. All writeup criteria are met by this document.
7. The submitted model is in an .h5  format and will run without errors.
[Rubric](https://review.udacity.com/#!/rubrics/1155/view)
 
---
### FIll Out Model_training.ipnyb

For the first attempt, code from the segmentation lab was added to the model training and any default parameters in the file are used. An Nvidia1080 graphics card was used instead of AWS

#### Encoder

The encoder function is used to filter the image data by increasing the depth of the image and decreasing the size of the image. An encoder will allow This can be applied several times until the image data is reduced to a 1x1 convolutional layer.

```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.    
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```


#### Decoder

The decoder function will reassemble an image after it has had the 1x1 convolution applied to it. The concatination allows skip connections to be applied to the network.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsample_sil = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output = layers.concatenate([upsample_sil, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output, filters)
    
    return output_layer
```

#### Fully Convolutional Network Model

The initial step was to test out an FCN with only 1 encoder and decoder layer each. This was to establish a baseline on how additional layers will affect the results of the FCN.

![FCN_fig_1](./writeup_images/First_FCN.png)

```ptyhon
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    x_encoded1 = encoder_block(inputs, 64, 2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    x_conv = conv2d_batchnorm(x_encoded1, 158)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    x = decoder_block(x_conv, inputs, 64)    
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

By only using a single encoder and decoder, not enough data was collected for a clear image. 3 layers will be tested out to gain more accurate images.



### Optimize Network and Parameters

For the initial test, the default parameters were kept as a baseline.
```python
learning_rate = 0.1
batch_size = 64
num_epochs = 1
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

Since the first training run took about 3 hours, the first parameter to improve was the number of workers. Brute force was utilized to determine the value, starting at 9657 seconds per epoch with two workers and then increasing the number of workers by 1 until the time per epoch began increasing again. It was quickly discovered that 2 was the fastest number with 10085 seconds at 1 worker and 9952 with 3 workers.

### Training the Network

With the initial architecture of one encoder, one decoder, and the default parameters the final score was about 10%.

### Future Enhancements



