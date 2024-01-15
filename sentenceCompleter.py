import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from transformer import Transformer
import sentencepiece as spm
import constants
import utilities

device = utilities.getDevice()
print(f"Device = {device}")

def generate(text):

  sp = spm.SentencePieceProcessor(model_file='tinystorycustom.model') #load pretrained tokeniser
  encodedSentence = torch.tensor(sp.encode_as_ids(text, add_bos=True)).long().unsqueeze(0).to(device) #tokenise input
  transformer = Transformer(constants.VOCAB_SIZE, constants.DIMENSIONS, constants.NUM_HEADS, constants.NUM_LAYERS, constants.D_FF, constants.MAX_SEQ_LENGTH, constants.DROPOUT).to(device) #instantiate decoder
  transformer.eval()
  utilities.load_latest_checkpoint(transformer)
  
  #generate 20 word output
  for _ in range(20):
    with torch.no_grad():
      logits = transformer(encodedSentence)
      logits = logits[:, -1, :] / 1.0
      probs = torch.nn.functional.softmax(logits, dim=-1)
      next = torch.multinomial(probs, num_samples=1)
      if next.item() == 2: break #end text generation if eos token produced
      encodedSentence = torch.cat([encodedSentence, next], dim=1)

  output = sp.decode(encodedSentence.tolist()[0])
  print(f"{text} - {output}")
  return { "story" : output } 

