import torch
from torch.utils.data import Dataset

class model_dataset(Dataset):
    
    def __init__(self ,tokens , blocksize):
        
        super().__init__()
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.long)
        self.tokens = tokens
        self.blocksize = blocksize
    
    def __len__(self):
        return len(self.tokens)-self.blocksize-1
    
    def __getitem__(self, i):
        
        x = self.tokens[i : i + self.blocksize]
        y = self.tokens[i + 1 : i + self.blocksize +1]
        
        return x,y
        


# x,y = next(iter(mydataset))

# print(x)
# print(y)


# count = 0

# for x ,y in mydataloader:
    
#     print(x[0])
#     print(y[0])
#     break
#     # count +=1 
#     # if count<20 :
       
    
    