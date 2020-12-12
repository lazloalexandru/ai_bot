- in the first epoch memory usage goes up from the 3Gb seen with modev_v10 to 18Gb,
altough the model is half the size. This may be cause due to the input size difference triggering
a different algorithm in the background! Cuda Convolution uses another approach.
- maybe input layer filer size 64 is too small. Test it!
Test Results
- after model beeing parametrized (see v12 model) realized that convolutional layer filters
need to be at least 128. If not huge performance decrease + memory usage jump.
- if using MaxPool also pay attention to double te size of filters on the next layer. Somehow this triggers some other algorithm in the background causeing performance drop. Further investigation neeeded, to clarify this.




