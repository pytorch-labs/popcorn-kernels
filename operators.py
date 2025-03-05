"""
List of PyTorch Operators

We will use this to combine the operators into synthetic programs.
"""

main_operators = ["Conv2d", "Matmul", "Gemm", "BMM", "Conv3d", "ConvTranspose2d", "ConvTranspose3d", "Conv1d", "ConvTranspose1d", "Linear", "Bilinear", "LazyLinear", "LazyConv1d", "LazyConv2d", "LazyConv3d", "LazyConvTranspose1d", "LazyConvTranspose2d", "LazyConvTranspose3d"]

activations = ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "GELU", "Swish", "Softmax", "Mish", "Hardtanh", "HardSwish", "ELU", "CELU", "SELU", "ReLU6", "PReLU", "RReLU", "LogSigmoid", "Softmin", "Softplus", "Softsign", "Softshrink", "Hardshrink", "Hardsigmoid", "Hardswish", "GLU", "SiLU", "Tanhshrink", "Threshold", "LogSoftmax", "Softmax2d"]
element_wise_ops = ["Add", "Multiply", "Subtract", "Divide", "Clamp", "Scale", "ResidualAdd", "Identity", "CosineSimilarity", "PairwiseDistance"]
normalizations = ["BatchNorm", "LayerNorm", "InstanceNorm", "GroupNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d", "LazyInstanceNorm1d", "LazyInstanceNorm2d", "LazyInstanceNorm3d", "LocalResponseNorm", "CrossMapLRN2d", "SyncBatchNorm", "RMSNorm"]
pooling = ["MaxPool", "AvgPool", "GlobalAvgPool", "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "FractionalMaxPool2d", "FractionalMaxPool3d", "LPPool1d", "LPPool2d", "LPPool3d", "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d"]
bias = ["BiasAdd"]
reductions = ["Sum", "Mean", "Max", "Min", "LogSumExp"]
regularization = ["Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "FeatureAlphaDropout"]
reshaping = ["Fold", "Unfold", "Flatten", "Unflatten"]
channel_ops = ["ChannelShuffle", "PixelShuffle", "PixelUnshuffle"]
padding = ["ZeroPad1d", "ZeroPad2d", "ZeroPad3d", "ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d", "CircularPad1d", "CircularPad2d", "CircularPad3d", "ConstantPad1d", "ConstantPad2d", "ConstantPad3d"]
upsampling = ["Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"]


# these are starting to be big, questionable???
embedding = ["Embedding", "EmbeddingBag"]
attention = ["MultiheadAttention"]
recurrent = ["RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell", "RNNBase", "RNNCellBase"]
transformer = ["Transformer", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer"]

# Update the all_operators list to include the new categories instead of "others"
all_operators = activations + element_wise_ops + normalizations + pooling + bias + reductions + regularization + reshaping + channel_ops + padding + embedding + recurrent + transformer + upsampling + attention




# Probably not good to have
containers = ["Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict", "Container", "Module"]
