from transformers import pipeline

prompt = "Hugging Face is a community-based open-source platform for machine learning."
generator = pipeline(task="text-generation")
print(generator(prompt))  # doctest: +SKIP
