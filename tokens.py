from datasets import load_dataset
import sentencepiece as spm
import constants

dataset = load_dataset("roneneldan/TinyStories")
model_prefix = "tinystorycustom"
vocab_size = constants.VOCAB_SIZE
text_file = 'tinystories'

with open(text_file,'w',encoding='utf-8') as file:
    for text in dataset['train']['text']:
        file.write(text)

# train SentencePiece tokeniser 
spm.SentencePieceTrainer.train(
    input=text_file,  # Pass the text data from the DataFrame
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    pad_id=3,
    input_sentence_size=constants.INPUT_SENTENCE_SIZE,
    shuffle_input_sentence=True
)
