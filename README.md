# Make a Bot Clone of You

The process of building, fine-tuning, and training a chatbot model using Microsoft's DialoGPT and your own iMessages. The steps include extracting messages from iMessages, processing the data, and fine-tuning the model on the extracted data.

Adapted from this article about Rick Sanchez Bot: https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30 

Before starting, make sure you have these: Python 3.8+, pip, SQLite3, Git
And make sure your terminal has Full Disk Access

Note: I recommend using a GPU for faster training and performance. I only have CPU and training took ages lol

If you're using a mac, go to System Preferences > Security & Privacy > Privacy > Full Disk Access on the side bar. Click the lock icon to make changes and check the 'Terminal' box. Click the lock icon again to save changes.

1) git clone 
cd 

2) Extract Your iMessages to CSV
The iMessages database is located at ~/Library/Messages/chat.db. To access this database, open terminal on your mac and run: sqlite3 ~/Library/Messages/chat.db. 

I extracted my message convos with one person by putting this SQL Query in the terminal:
```
.headers on
.mode csv
.output messages_history.csv
--replace 'messages_history.csv' with whatever you want to call your csv file
--good to put the path to messages_history.csv file
SELECT
    datetime(message.date / 1000000000 + strftime('%s', '2001-01-01'), 'unixepoch', 'localtime') AS date,
    CASE
        WHEN message.is_from_me = 1 THEN 'Me'
        --if the message is from me, then the column should have 'Me'
        ELSE handle.id
    END AS sender_name,
    message.text AS content
FROM
    message
    JOIN handle ON message.handle_id = handle.ROWID
WHERE
    handle.id = '<CONTACT_NUMBER>'
    --replace CONTACT_NUMBER with the contact number of the person whose messages you want to retrieve. Note: should begin with +(code)(rest of number)
ORDER BY
    message.date;
```
Open 'messages_history.csv' or the csv with the data retrieved and double check that there are date, sender_name, and content columns.

3) Create a virtual environment and install the required libraries.
```
python -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn torch transformers tqdm tensorboardX
```

4) Run the prepare_data.py script to preprocess and split the data:
```
python prepare_data.py
```
This script will generate train.csv and val.csv files from messages_history.csv

If you are using a GPU and want to enable fp16 with Apex:
- Install Apex
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir ./
```
- Set 'self.fp16 = True' in train.py

5) Run the train.py script to fine-tune the model:
```
python train.py
```

# How it works
The chatbot in this project is fine-tuned using the DialoGPT model, which is a variant of GPT-2 (Generative Pre-trained Transformer 2). The model is pre-trained on a large dataset of conversational data, making it suitable for generating human-like responses in a dialogue context. Here's a detailed explanation of how the chatbot works and its architecture:

GPT-2 is based on the Transformer architecture, and DialoGPT is a variant of GPT-2 specifically fine-tuned on a large dataset of dialogue interactions.

* Data Extraction: The chat data is extracted from the iMessages database (chat.db) using an SQL query. The data includes columns for date, sender name, and message content. The extracted data is saved in a CSV file (messages_history.csv) for further processing.
* Pre-processing: prepare_data.py processes the raw data and splits it into training (train.csv) and validation (val.csv) datasets.
* Tokenizer: The input text is tokenized using the AutoTokenizer from the Hugging Face Transformers library. This converts the text into numerical tokens that the model can process.
* ConversationDataset: This class constructs input sequences by concatenating previous dialogue turns with the current response.
* Training Loop: The train.py script handles the fine-tuning process. It includes a training loop that iterates over the dataset, updating the model weights to minimize the loss.
* Optimization: The AdamW optimizer and a learning rate scheduler are used to optimize the model parameters. Gradient accumulation is employed to handle large batches that may not fit into memory.
* Validation: The model's performance is evaluated on the validation dataset at regular intervals during training. We use loss and perplexity to model progress.
Checkpoints: Model checkpoints are saved periodically.

After this is done, we test by generating responses. We can use beam search to improve the quality of the generated responses by considering multiple potential outputs.
Next, the generated token sequence is decoded back into text using the tokenizer. The chatbot can then output this text as its response.
Once fine-tuning is complete, the model can be used to generate responses in a chatbot application. The trained model can understand and generate text with emojis.
