from datasets import load_dataset
import sentencepiece as spm

# train_ds =load_dataset('Salesforce/wikitext',"wikitext-2-v1" ,split="train")
# val_ds =load_dataset('Salesforce/wikitext',"wikitext-2-v1" ,split="validation")
# test_ds =load_dataset('Salesforce/wikitext',"wikitext-2-v1" ,split="test")

# print(train_ds.shape)
# print(train_ds[1200])
# print(val_ds.shape)
# print(test_ds.shape)


# with open('Mini_gpt/model_dataset/train_dataset.txt',mode="w" ,encoding='utf-8') as f:
#     for row in train_ds:
#         text = row["text"].strip()
#         f.write(text + '\n')
        

# with open('Mini_gpt/model_dataset/val_dataset.txt',mode="w" ,encoding='utf-8') as f:
#     for row in val_ds:
#         text = row["text"].strip()
#         f.write(text + '\n')
        

# with open('Mini_gpt/model_dataset/test_dataset.txt',mode="w" ,encoding='utf-8') as f:
#     for row in test_ds:
        # text = row["text"].strip()
        # f.write(text + '\n')
        
with open('Mini_gpt/model_dataset/train_short.txt' , mode='r' , encoding='utf-8') as f :
        train_dataset = f.read()
        


# spm.SentencePieceTrainer.Train(input = "Mini_gpt/model_dataset/train_dataset.txt",
#                                model_prefix ='Mini_gpt/tokenizer_files/mini_gpt_tokenizer',
#                                vocab_size = 16000,
#                                model_type ='bpe')


sp = spm.SentencePieceProcessor()
sp.load('Mini_gpt/tokenizer_files/mini_gpt_tokenizer.model')
vocab_size =sp.GetPieceSize()


# Text to ids function 

def encoding(text):
    
    input_ids = sp.Encode(text,out_type=int)
    return input_ids
    

train_input_ids = encoding(train_dataset)
# print(train_input_ids[:30])

 