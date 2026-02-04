import torch
from torch.utils.data import Dataset,DataLoader
from preprocessing import train_input_ids



class model_dataset(Dataset):
    
    def __init__(self ,tokens , blocksize):
        
        super().__init__()
        
        self.tokens = tokens
        self.blocksize = blocksize
    
    def __len__(self):
        return len(self.tokens)-self.blocksize-1
    
    def __getitem__(self, i):
        
        x = self.tokens[i : i + self.blocksize]
        y = self.tokens[i + 1 : i + self.blocksize +1]
        
        return x,y
        

mydataset = model_dataset(train_input_ids ,blocksize=8)

# x,y = next(iter(mydataset))

# print(x)
# print(y)


mydataloader = DataLoader(dataset=mydataset , batch_size=32 ,shuffle=True)

# count = 0

# for x ,y in mydataloader:
    
#     print(x[0])
#     print(y[0])
#     break
#     # count +=1 
#     # if count<20 :
       
    
    