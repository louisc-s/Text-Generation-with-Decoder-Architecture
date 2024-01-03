# Text-Generation-with-Decoder-Architecture
Code to implemnent a decoder only transformer model that predicts the next sentence from a given input phrase

## Overview

This code implemented a decoder only transformer model with a multihead attention mechanism. This model was trained using the TinyStories dataset and produces a 20 word generative output which follows on semantically and syntactically from a given input phrase. 

## Project Structure 

1. tokens.py - creates and trains sentence piece tokeniser
  
2. dataset.py -  implements tokenised dataset
   
3. positional_encoding.py - implements positional encoding from decoder architecture 
   
4. multi_head_attention.py - implements multihead attention mechansim from decoder architecture
   
5. position_wise_feed_forward.py - implements position wise feedforward layers from from decoder architecture

6. decoder_layer.py - implements decoder layer from previosuly defined building blocks

7. transformer.py - implements decoder only transformer from decoder layer and positional encoder building blocks

8. train.py - trains the transformer using TinyStories dataset

9. sentence_completer.py - generates output text from input phrase

10. server.py - connects to server to allow model to be accessed and interacted with from website

11. constants.py - contains constants for the project

12. utilities.py - contains simple functions for accessing and loading latest transformer models
    
## Author 

Louis Chapo-Saunders
