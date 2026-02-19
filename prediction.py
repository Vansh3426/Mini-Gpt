
import torch
from model import Decoder_block
from preprocessing import sp ,vocab_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 16

model = Decoder_block(vocab_size , embed_dim=64 , n_head=4).to(device)
model.load_state_dict(torch.load('Mini_gpt/saved_model_and_files/trained_model_full_dataset.pth'))


model.eval()
maxlength = 50
text = ' what is the  '


def generate(text, maxlength):
    
    ids = sp.Encode(text , out_type= int)
        
    decoder_input =torch.tensor( ids ,device=device).unsqueeze(0)
    print(decoder_input.shape)
        
    for _ in range(maxlength-1):
        
        idx_cond = decoder_input[:, -block_size:]
        logits = model(idx_cond)
        
        logits= logits[: ,-1 ,:]
            
        probs = torch.softmax(logits, dim=-1)
        
        next_token_tensor = torch.multinomial(probs ,1)
         
        # next_token_tensor = torch.argmax(next_token , dim=-1).unsqueeze(dim=1)       
        # print(next_token_tensor.shape)
        
        decoder_input= torch.cat( (decoder_input , next_token_tensor),dim=1)
    
    token_ids = decoder_input[0].tolist()
    
    return sp.Decode(token_ids)
        
        
        
output = generate(text , maxlength)

print(output)
        

    
