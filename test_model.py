from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer from the checkpoint
# output is the directory where the model is saved after training
model_path = 'output/checkpoint-1000'
try:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except OSError as e:
    print(f"Error loading model: {e}")
    exit(1)

# Move the model to the correct device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def chat_with_bot():
    chat_history_ids = None
    print("Start chatting with the bot (type 'exit' to stop)...")
    while True:
        user_input = input(">> User: ")
        if user_input.lower() == 'exit':
            break
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids
        
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=500,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {bot_response}")

if __name__ == "__main__":
    chat_with_bot()