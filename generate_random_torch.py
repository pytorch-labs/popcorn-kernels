from openai import OpenAI

import random
import tomli
import dotenv
import uuid
import argparse
import os
import tqdm
import concurrent.futures
from typing import List

torch_nn_modules = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveLogSoftmaxWithLoss",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AlphaDropout",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BCELoss",
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CELU",
    "CTCLoss",
    "ChannelShuffle",
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Container",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CosineEmbeddingLoss",
    "CosineSimilarity",
    "CrossEntropyLoss",
    "CrossMapLRN2d",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "FeatureAlphaDropout",
    "Flatten",
    "Fold",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "GELU",
    "GLU",
    "GRU",
    "GRUCell",
    "GaussianNLLLoss",
    "GroupNorm",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "Identity",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "KLDivLoss",
    "L1Loss",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "LSTM",
    "LSTMCell",
    "LayerNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LazyLinear",
    "LeakyReLU",
    "Linear",
    "LocalResponseNorm",
    "LogSigmoid",
    "LogSoftmax",
    "MSELoss",
    "MarginRankingLoss",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "Mish",
    "Module",
    "ModuleDict",
    "ModuleList",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "MultiheadAttention",
    "NLLLoss",
    "NLLLoss2d",
    "PReLU",
    "PairwiseDistance",
    "ParameterDict",
    "ParameterList",
    "PixelShuffle",
    "PixelUnshuffle",
    "PoissonNLLLoss",
    "RMSNorm",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
    "RReLU",
    "ReLU",
    "ReLU6",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "SyncBatchNorm",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]

def generate_prompt(num_modules: int = 10):
    assert num_modules <= len(torch_nn_modules)
    random_modules = random.sample(torch_nn_modules, num_modules)
    with open("prompts.toml", "rb") as f:
        prompt_dict = tomli.load(f)['prompts']
    prompt_dict = {prompt['name']: prompt for prompt in prompt_dict}
    prompt = prompt_dict["generate_random_torch"]["prompt"].replace("{{modules}}", str(random_modules))
    return prompt

def generate_random_torch_from_prompt(prompt: str, model_name="gpt-4o-mini"):
    if "gpt" in model_name or "o1" in model_name:
        client = OpenAI()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def extract_code(response: str):
    code = response.split("```python")[1].split("```")[0]
    return code

def write_code_to_file(code: str):
    filename = f"generated/random_torch_{uuid.uuid4()}.py"
    with open(filename, "w") as f:
        f.write(code)

def generate_single_model(model_name: str) -> None:
    num_modules = random.randint(3, 10)
    prompt = generate_prompt(num_modules)
    code = extract_code(generate_random_torch_from_prompt(prompt, model_name))
    write_code_to_file(code)

def generate_random_torch_model(num_models: int = 1, model_name: str = "deepseek-chat", max_workers: int = 50):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = [
            executor.submit(generate_single_model, model_name)
            for _ in range(num_models)
        ]
        
        # Show progress bar for completed futures
        for _ in tqdm.tqdm(
            concurrent.futures.as_completed(futures), 
            total=num_models,
            desc="Generating models"
        ):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--max_workers", type=int, default=50)
    args = parser.parse_args()
    dotenv.load_dotenv()
    generate_random_torch_model(args.num_models, args.model_name, args.max_workers)

