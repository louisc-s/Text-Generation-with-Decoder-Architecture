import torch
from datasets import load_dataset
import sentencepiece as spm

#create tokenised dataset 
class TinyStoriesData(torch.utils.data.Dataset):
  def __init__(self, name, mode, max_seq_length):
    self.dataset = load_dataset(name,split=mode)
    self.sp = spm.SentencePieceProcessor(model_file='tinystorycustom.model')
    
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sentence = self.dataset[idx]["text"]
    encoded =self.sp.encode_as_ids(sentence, add_bos=True, add_eos=True)
    return torch.tensor(encoded)
  
  def collate_function(self, batch):
    return torch.nn.utils.rnn.pad_sequence([item for item in batch], batch_first=True, padding_value=3)
