from transformers import pipeline

prompt = "Hugging Face is a community-based open-source platform for machine learning."
generator = pipeline(
    task="text-generation", model="/var/wd_smit/localdata/models/Llama-2-7b-chat-hf"
)
print(generator(prompt))  # doctest: +SKIP
