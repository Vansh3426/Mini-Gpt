import torch 
from torch import nn
from model_dataset_class import model_dataset
from torch.utils.data import DataLoader

vocab_size =16000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == "cuda":
    torch.zeros(1, device=device)
    
torch.cuda.manual_seed(42)
torch.manual_seed(42)

block_size = 16

print("Script started")

class Masked_multihead_attention(nn.Module):
    
    def __init__(self , vocab_size , embed_dim ,n_head  ,dropout):
        super().__init__()

        self.embed = embed_dim
        self.embed = nn.Embedding(vocab_size , embed_dim )
        self.n_head =n_head
        self.head_size = embed_dim // self.n_head 
        
        self.positional_embed = nn.Embedding( block_size , embed_dim)
        self.q = nn.Linear( embed_dim , embed_dim )
        self.k = nn.Linear( embed_dim ,embed_dim )
        self.v = nn.Linear( embed_dim ,embed_dim )
        
        self.dropout =nn.Dropout(dropout)
        self.register_buffer('tril' ,torch.tril(torch.ones(block_size , block_size)))    
    
    def forward(self, x ):
        
        B , T  = x.shape
        
        x_embed = self.embed(x)
       
        pos = torch.arange(T ,device=x.device)
        x_pos = self.positional_embed(pos)
        x_embeddings = x_embed + x_pos
    
 
        query =self.q(x_embeddings)
        key =self.k(x_embeddings)
        value =self.v(x_embeddings)
          
        # print(key.shape)
        # print(query.shape)
        # print(value.shape)
        
        query =query.view(B , T , self.n_head ,self.head_size).transpose(1,2)
        key =key.view(B , T , self.n_head ,self.head_size).transpose(1,2)
        value =value.view(B , T , self.n_head ,self.head_size).transpose(1,2)
        # print(key.shape)
        # print(query.shape)
        # print(value.shape)
        
        
        wei = query @ key.transpose(-2,-1)
        # print(wei.shape)
        
  
        
        wei = wei.masked_fill(self.tril[:T ,:T] == 0 , float('-inf'))
        
        wei = wei /(self.head_size ** 0.5)
        
        softmax_wei = torch.softmax(wei , dim=-1)
        dropout_wei = self.dropout(softmax_wei)
        
        final_wei = dropout_wei @ value
        
        out = final_wei.transpose(1, 2).contiguous().view(B, T, -1)

        # print(out.shape)
        
        return out
    
    
    
class Feed_forward(nn.Module):
    
    def __init__(self , embed_dim ,dropout ):
        super().__init__()
        
        self.layer =nn.Sequential(
            nn.Linear(in_features = embed_dim , out_features= 2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features = 2 * embed_dim , out_features= embed_dim),
            nn.Dropout(dropout)
            
        )
        
        
    def forward(self ,x ):
        
        x = self.layer(x)

        
        return x 
        
        

class Layer_norm(nn.Module):
    
    def __init__(self , embedding_dim , eps =1e-5):
        
        super().__init__()
        
        self.gamma =nn.Parameter(torch.ones(embedding_dim))
        self.beta =nn.Parameter(torch.zeros(embedding_dim))
        self.eps =eps
        
    def forward(self , x):
        
        # x.shape = batch_size , seq_length , embedding_dim 
        
        
        mean = x.mean(dim=-1 , keepdim =True)
        var = x.var(dim=-1 ,keepdim=True ,unbiased=False)
        
        x_hat = (x - mean)/torch.sqrt(var + self.eps)
        
        x_norm = self.gamma * x_hat + self.beta
        
        return x_norm
    
        
    
class Decoder_block(nn.Module):
    
    def __init__(self ,vocab_size , embed_dim ,n_head ,dropout =0.1):
        super().__init__()
        
        self.attention = Masked_multihead_attention(vocab_size,embed_dim,n_head,dropout)
        self.norm = Layer_norm(embedding_dim=embed_dim)
        self.ff = Feed_forward(embed_dim,dropout)
        self.linear =nn.Linear(embed_dim ,vocab_size)
    def forward(self , x):
        
        x = self.attention(x)
        x  = self.norm(x)
        x = x + self.ff(x)
        x = self.linear(x)
        
        return x
        
     

if __name__ == '__main__':
        
    model = Decoder_block(vocab_size , embed_dim=64 , n_head=4).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters() , lr = 0.0003 , weight_decay=0.01)
    
    train_input_ids = torch.load("Mini_gpt/saved_tokens/train_ids.pt")
    val_input_ids = torch.load("Mini_gpt/saved_tokens/val_ids.pt")

    
    mydataset = model_dataset(train_input_ids ,block_size)
    myvaldataset =model_dataset(val_input_ids ,block_size)
    
    mydataloader = DataLoader(dataset=mydataset , batch_size=512 ,shuffle=True ,pin_memory=True)
    valdataloader =DataLoader(dataset=myvaldataset ,batch_size=512 ,pin_memory=True)

    
    epochs = 20
    best_loss =float('inf')

    for epoch in range(epochs):
        
        model.train()

        total_loss =  0
        
        
        for x,y in mydataloader:
            
            x , y = x.to(device) , y.to(device)
            
            logits = model(x)

            optimizer.zero_grad()
            
            loss = loss_fn(logits.reshape(-1,logits.size(-1)) ,y.reshape(-1))
            
            total_loss += loss.item()
            
            
            loss.backward()
            
            optimizer.step()
              
        
                
        model.eval()

        with torch.inference_mode():
            

            total_val_loss =  0
            
            for x1,y1 in valdataloader:
                
                x1 , y1 = x1.to(device) , y1.to(device)
                
                logits = model(x1)
                
                loss = loss_fn(logits.reshape(-1,logits.size(-1)) ,y1.reshape(-1))
                
                total_val_loss += loss.item()
                
            avg_loss = total_val_loss / len(valdataloader)

            if avg_loss < best_loss:
                best_loss =avg_loss
                torch.save(model.state_dict() ,'Mini_gpt/saved_model_and_files/trained_model_full_dataset.pth')
              
                    
        print(f' Epoch :{epoch}     |     loss : {total_loss/len(mydataloader)}    val loss : {total_val_loss/len(valdataloader)}')    
        
            