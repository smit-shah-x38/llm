from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "lmsys/vicuna-7b-v1.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
    "Describe the character of the godfather Vito Corleone. Answer: ",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.5,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
