from datasets import load_dataset
import sentencepiece as spm

train_ds =load_dataset('Salesforce/wikitext',"wikitext-2-v1" ,split="train")
# val_ds =load_dataset('Salesforce/wikitext',"wikitext-2-v1" ,split="validation")
# test_ds =load_dataset('Salesforce/wikitext',"wikitext-2-v1" ,split="test")

# print(train_ds.shape)
# print(train_ds[1200])
# print(val_ds.shape)
# print(test_ds.shape)


# with open('Mini_gpt/dataset/train_dataset.txt',mode="w" ,encoding='utf-8') as f:
#     for row in train_ds:
#         text = row["text"].strip()
#         f.write(text + '\n')
        

# with open('Mini_gpt/dataset/val_dataset.txt',mode="w" ,encoding='utf-8') as f:
#     for row in val_ds:
#         text = row["text"].strip()
#         f.write(text + '\n')
        

# with open('Mini_gpt/dataset/test_dataset.txt',mode="w" ,encoding='utf-8') as f:
#     for row in test_ds:
        # text = row["text"].strip()
        # f.write(text + '\n')
        
    
        

# spm.SentencePieceTrainer.Train(input = "Mini_gpt/dataset/train_dataset.txt",
#                                model_prefix ='Mini_gpt/tokenizer_files/mini_gpt_tokenizer',
#                                vocab_size = 16000,
#                                model_type ='bpe')


sp = spm.SentencePieceProcessor()
sp.load('Mini_gpt/tokenizer_files/mini_gpt_tokenizer.model')
vocab_size =sp.GetPieceSize()


# Text to ids function 

def encoding(row):
    
    input_ids = sp.Encode(row['text'],out_type=int)
    