from openai import OpenAI

client = OpenAI()
import random
import tomli
import dotenv
import uuid

# torch_nn_modules = [
#     "AdaptiveAvgPool1d",
#     "AdaptiveAvgPool2d",
#     "AdaptiveAvgPool3d",
#     "AdaptiveLogSoftmaxWithLoss",
#     "AdaptiveMaxPool1d",
#     "AdaptiveMaxPool2d",
#     "AdaptiveMaxPool3d",
#     "AlphaDropout",
#     "AvgPool1d",
#     "AvgPool2d",
#     "AvgPool3d",
#     "BCELoss",
#     "BCEWithLogitsLoss",
#     "BatchNorm1d",
#     "BatchNorm2d",
#     "BatchNorm3d",
#     "Bilinear",
#     "CELU",
#     "CTCLoss",
#     "ChannelShuffle",
#     "CircularPad1d",
#     "CircularPad2d",
#     "CircularPad3d",
#     "ConstantPad1d",
#     "ConstantPad2d",
#     "ConstantPad3d",
#     "Container",
#     "Conv1d",
#     "Conv2d",
#     "Conv3d",
#     "ConvTranspose1d",
#     "ConvTranspose2d",
#     "ConvTranspose3d",
#     "CosineEmbeddingLoss",
#     "CosineSimilarity",
#     "CrossEntropyLoss",
#     "CrossMapLRN2d",
#     "Dropout",
#     "Dropout1d",
#     "Dropout2d",
#     "Dropout3d",
#     "ELU",
#     "Embedding",
#     "EmbeddingBag",
#     "FeatureAlphaDropout",
#     "Flatten",
#     "Fold",
#     "FractionalMaxPool2d",
#     "FractionalMaxPool3d",
#     "GELU",
#     "GLU",
#     "GRU",
#     "GRUCell",
#     "GaussianNLLLoss",
#     "GroupNorm",
#     "Hardshrink",
#     "Hardsigmoid",
#     "Hardswish",
#     "Hardtanh",
#     "HingeEmbeddingLoss",
#     "HuberLoss",
#     "Identity",
#     "InstanceNorm1d",
#     "InstanceNorm2d",
#     "InstanceNorm3d",
#     "KLDivLoss",
#     "L1Loss",
#     "LPPool1d",
#     "LPPool2d",
#     "LPPool3d",
#     "LSTM",
#     "LSTMCell",
#     "LayerNorm",
#     "LazyBatchNorm1d",
#     "LazyBatchNorm2d",
#     "LazyBatchNorm3d",
#     "LazyConv1d",
#     "LazyConv2d",
#     "LazyConv3d",
#     "LazyConvTranspose1d",
#     "LazyConvTranspose2d",
#     "LazyConvTranspose3d",
#     "LazyInstanceNorm1d",
#     "LazyInstanceNorm2d",
#     "LazyInstanceNorm3d",
#     "LazyLinear",
#     "LeakyReLU",
#     "Linear",
#     "LocalResponseNorm",
#     "LogSigmoid",
#     "LogSoftmax",
#     "MSELoss",
#     "MarginRankingLoss",
#     "MaxPool1d",
#     "MaxPool2d",
#     "MaxPool3d",
#     "MaxUnpool1d",
#     "MaxUnpool2d",
#     "MaxUnpool3d",
#     "Mish",
#     "Module",
#     "ModuleDict",
#     "ModuleList",
#     "MultiLabelMarginLoss",
#     "MultiLabelSoftMarginLoss",
#     "MultiMarginLoss",
#     "MultiheadAttention",
#     "NLLLoss",
#     "NLLLoss2d",
#     "PReLU",
#     "PairwiseDistance",
#     "ParameterDict",
#     "ParameterList",
#     "PixelShuffle",
#     "PixelUnshuffle",
#     "PoissonNLLLoss",
#     "RMSNorm",
#     "RNN",
#     "RNNBase",
#     "RNNCell",
#     "RNNCellBase",
#     "RReLU",
#     "ReLU",
#     "ReLU6",
#     "ReflectionPad1d",
#     "ReflectionPad2d",
#     "ReflectionPad3d",
#     "ReplicationPad1d",
#     "ReplicationPad2d",
#     "ReplicationPad3d",
#     "SELU",
#     "Sequential",
#     "SiLU",
#     "Sigmoid",
#     "SmoothL1Loss",
#     "SoftMarginLoss",
#     "Softmax",
#     "Softmax2d",
#     "Softmin",
#     "Softplus",
#     "Softshrink",
#     "Softsign",
#     "SyncBatchNorm",
#     "Tanh",
#     "Tanhshrink",
#     "Threshold",
#     "Transformer",
#     "TransformerDecoder",
#     "TransformerDecoderLayer",
#     "TransformerEncoder",
#     "TransformerEncoderLayer",
#     "TripletMarginLoss",
#     "TripletMarginWithDistanceLoss",
#     "Unflatten",
#     "Unfold",
#     "Upsample",
#     "UpsamplingBilinear2d",
#     "UpsamplingNearest2d",
#     "ZeroPad1d",
#     "ZeroPad2d",
#     "ZeroPad3d",
# ]

torch_nn_modules = [
    "Conv2d",
    "MaxPool2d",
    "Linear",
    "LogSoftmax",
    "ReLU",
]

def generate_prompt(num_modules: int = 5):
    assert num_modules <= len(torch_nn_modules)
    random_modules = random.sample(torch_nn_modules, num_modules)
    with open("prompts.toml", "rb") as f:
        prompt_dict = tomli.load(f)['prompts']
    prompt_dict = {prompt['name']: prompt for prompt in prompt_dict}
    prompt = prompt_dict["generate_random_torch"]["prompt"].replace("{{modules}}", str(random_modules))
    return prompt

def generate_random_torch_from_prompt(prompt: str):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def extract_code(response: str):
    code = response.split("```python")[1].split("```")[0]
    return code

def write_code_to_file(code: str):
    filename = f"generated/random_torch_{uuid.uuid4()}.py"
    with open(filename, "w") as f:
        f.write(code)

def generate_random_torch_model(num_models: int = 1):
    for _ in range(num_models):
        num_modules = random.randint(3, 5)
        prompt = generate_prompt(num_modules)
        code = extract_code(generate_random_torch_from_prompt(prompt))
        write_code_to_file(code)

if __name__ == "__main__":
    dotenv.load_dotenv()
    generate_random_torch_model(10)

