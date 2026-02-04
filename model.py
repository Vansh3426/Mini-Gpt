import torch 
from torch import nn
from model_dataset import mydataloader

block_size = 8
vocab_size = 1000

torch.manual_seed(42)

class Masked_multihead_attention(nn.Module):
    
    def __init__(self , vocab_size , embed_dim ,n_head ):
        super().__init__()

        self.embed = embed_dim
        self.embed = nn.Embedding(vocab_size , embed_dim )
        self.n_head =n_head
        self.head_size = embed_dim // self.n_head 
        
        self.positional_embed = nn.Embedding( block_size , embed_dim)
        self.q = nn.Linear( embed_dim , embed_dim )
        self.k = nn.Linear( embed_dim ,embed_dim )
        self.v = nn.Linear( embed_dim ,embed_dim )
        
        self.register_buffer('tril' ,torch.tril(torch.ones(block_size , block_size)))    
    
    def forward(self, x ):
        
        B , T  = x.shape
        
        x_embed = self.embed(x)
        x_pos = self.positional_embed(torch.arange(T))
        x_embeddings = x_embed + x_pos
    
 
        query =self.q(x_embeddings)
        key =self.k(x_embeddings)
        value =self.v(x_embeddings)
          
        print(key.shape)
        print(query.shape)
        print(value.shape)
        
        query =query.view(B , T , self.n_head ,self.head_size).transpose(1,2)
        key =key.view(B , T , self.n_head ,self.head_size).transpose(1,2)
        value =value.view(B , T , self.n_head ,self.head_size).transpose(1,2)
        print(key.shape)
        print(query.shape)
        print(value.shape)
        
        
        wei = query @ key.transpose(-2,-1)
        print(wei.shape)
        
  
        
        wei = wei.masked_fill(self.tril[:T ,:T] == 0 , float('-inf'))
        
        wei = wei / torch.sqrt(torch.tensor(self.head_size))
        
        softmax_wei = torch.softmax(wei , dim=-1)
        
        final_wei = softmax_wei @ value
        
        out = final_wei.transpose(1, 2).contiguous().view(B, T, -1)

        print(out.shape)
        
        return out
    
# a = torch.tril(torch.ones(3,3))
# a = a / torch.sum(a ,1 , keepdim =True)
# b = torch.randint(0,10,(3,3)).float()

# c = a @ b


# print(a)
# print(b)
# print(c)



# trill = torch.tril(torch.ones(3,3))
# wei = torch.zeros(3,3)
# wei= wei.masked_fill(trill == 0 , float('-inf'))
# new = torch.softmax(wei , dim =-1)

# x = torch.randint(0,10,(3,3)).float()

# output = new @ x



# print(new)
# print(output)

x = torch.randint(1,10,(1 ,8))

model = Masked_multihead_attention(vocab_size , embed_dim=32 , n_head=4)

output = model(x)