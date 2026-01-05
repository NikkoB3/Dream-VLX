# Dream-VL
## Installation

```bash
# Create and activate conda environment
conda create -n dreamvl python=3.10 -y
conda activate dreamvl

# Install PyTorch
# Use a comamand specific to your machine: https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

cd train
pip install -e ".[train]"

cd ../eval
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install "flash-attn==2.5.5" --no-build-isolation # do the following if `import flash_attn` failed
# pip install packaging ninja
# ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention
# python setup.py install
```

`requirements.txt` is the full used environment for reference.

## Quict Start

Below shows an example of how to use Dream-VL.

```python
import torch
from transformers import AutoProcessor, AutoModel

model_name = "Dream-org/Dream-VL-7B"

# from dreamvl.modeling_dreamvl import DreamVLModel
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to('cuda')

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

####### Method 1
from PIL import Image
import requests
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
messages = [
    {
        "role": "user","content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}]
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(text)
inputs = processor(
    text=[text], images=[image], padding=True, return_tensors="pt"
)

####### Method 2: use qwen_vl_utils
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# from qwen_vl_utils import process_vision_info
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

inputs = inputs.to("cuda")
input_ids = inputs.pop("input_ids")
output = model.diffusion_generate(
    input_ids,
    max_new_tokens=128,
    output_history=True,
    return_dict_in_generate=True,
    steps=128,
    temperature=0.1,
    top_p=1,
    alg="maskgit_plus",
    alg_temp=0,
    use_cache=False,
    **inputs
)

generations = [
    processor.tokenizer.decode(g[len(p):].cpu().tolist())
    for p, g in zip(input_ids, output.sequences)
]

for j in range(len(messages)):
    print("output:", j, generations[j].split(processor.tokenizer.eos_token)[0])

# output: The image depicts a serene beach scene featuring a young woman and a golden retriever. The woman, dressed in a plaid shirt and dark pants, is seated on the sandy shore, smiling warmly at the camera. The golden retriever, adorned with a colorful harness, sits attentively beside her, its gaze fixed on the woman. The background reveals the vast expanse of the ocean, with waves gently kissing the shore. The sky above is a clear blue, suggesting a sunny day. The overall atmosphere exudes a sense of peace and companionship between the woman and her dog.
```

## Training

### Sample Data and Format

Here is an example of training data:

```json
{
   "id": str,
   "image": str/array,
   "video": str,
   "conversations": array,
}
```
<!-- ![ex](data/images/cultural/2433684022797.0.jpg)

The corresponding image file for this example is located at `data/images/cultural/2433684022797.0.jpg`. -->

Data Structure:
- **id**: Unique identifier for the data sample.
- **image**: The path to the image file used in this instance.
- **video**: The path to the video file used in this instance.
- **conversations**: A series of conversations between the "human" and the model (in this case, referred to as "gpt").
   - **from**: Identifies the speaker (either "human" or "gpt").
   - **value**: The content of the message, which can include both text and image references.
<!-- - **language**: The language of the instruction and conversation (in this example, it is Korean). -->


### Stage 1: Pretraining

After setting up, initiate the pretraining phase:

1. **Run the Pretraining Script**:

```bash
cd train

bash scripts/train/dream_vl/pretrain.sh
```
This result in the creation of a `mm_projector.bin` file essential for the finetuning stage.

Once pretraining is complete, proceed to finetune the model: **Ensure Fine-tuning Data is Available**

### Stage 2: Fine-tuning(SI)

After obtaining the single-image data, run the following script to begin fine-tuning:

```bash
cd train

# 1 ep with lr 1e-5
bash scripts/train/dream_vl/finetune_si.sh

# 2 ep with lr 5e-6
bash scripts/train/dream_vl/finetune_si2.sh
```

### Stage 3: Fine-tuning(OV)

After obtaining the ov data, run the following script to begin fine-tuning:

```bash
cd train

bash scripts/train/dream_vl/finetune_ov.sh
```

### Convert and Upload Models
After trained the model, use `python export.py` to convert and upload the trained model to HuggingFace.

## Evaluation

To evaluate the model's capabilities:

1. **Navigate to the Evaluation Directory**:

```bash
cd eval
```

2. **Run the Evaluation Script**:

To run the evaluation, use the following command:
```bash
bash eval_dream_vl.sh
```

If you want to eval a checkpoint during training (i.e., before running `python export.py`), use:
```bash
bash eval_llava_dream_vl.sh
```



## Acknowledgement
The training code is built upon [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [MAmmoTH-VL](https://github.com/MAmmoTH-VL/MAmmoTH-VL), we would like to thank them for the helpful code!
