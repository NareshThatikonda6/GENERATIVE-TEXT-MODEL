from transformers import pipeline

print("Loading GPT-2 model...")

# Load pretrained GPT-2 model
generator = pipeline("text-generation", model="gpt2")

print("Model loaded successfully!")

# Function to generate text
def generate_text(prompt):

    result = generator(
    prompt,
    max_new_tokens=120,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=50256
)

    generated_text = result[0]["generated_text"]

    print("\n--------------------------------------")
    print("User Prompt:")
    print(prompt)

    print("\nGenerated Paragraph:")
    print(generated_text)
    print("--------------------------------------")

# Example prompts
generate_text("Artificial Intelligence is transforming industries because")

generate_text("The future of software engineering will depend on")

generate_text("The future of software engineering will depend on")

# User input prompt
user_prompt = input("\nEnter your topic or sentence: ")

generate_text(user_prompt)

print("\nText generation completed successfully!")