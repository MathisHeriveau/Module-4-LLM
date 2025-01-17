from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define the input text
text = "1+1= ?"

# Encode the input text
encoded_input = tokenizer(text, return_tensors='pt')


output = model.generate(
    **encoded_input,
    max_length=50,  
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    no_repeat_ngram_size=2
)



# Decode the generated tokens back to text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the decoded text
print("decode " + decoded_output)
print("enddecode ")
