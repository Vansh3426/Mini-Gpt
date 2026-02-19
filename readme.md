▰ Mini GPT – Transformer Decoder Language Model ▰

A from-scratch implementation of a GPT-style decoder-only Transformer built in PyTorch, trained on the WikiText-2 dataset using a custom
SentencePiece tokenizer.

This project demonstrates a complete language modeling pipeline including:

∘ Dataset preprocessing
∘ Custom BPE tokenizer (SentencePiece)
∘ Masked Multi-Head Attention
∘ Transformer Decoder Block
∘ Custom Layer Normalization
∘ Training & Validation loop
∘ Checkpoint saving
∘ Text generation pipeline

📌 Dataset

∘ Dataset: Salesforce/wikitext – wikitext-2-v1

∘ Loaded using HuggingFace:

 load_dataset('Salesforce/wikitext', "wikitext-2-v1")

Token Statistics

∘ Total Train Tokens: 2,237,611
∘ Total Validation Tokens: 222,858
∘ Vocabulary Size: 16,000 (BPE – SentencePiece)

🏗 Model Architecture

🔹 Type
∘ Decoder-only Transformer (GPT-style)

🔹 Hyperparameters

∘ Parameter	Value
∘ Embedding Dimension	64
∘ Heads	4
∘ Block Size	16
∘ Vocabulary Size	16,000
∘ Dropout	0.1
∘ Optimizer	AdamW
∘ Learning Rate	3e-4
∘ Weight Decay	0.01
∘ Batch Size	512
∘ Epochs	20

🧩 Model Components

1️⃣ Masked Multi-Head Self Attention

∘ Learned token embeddings
∘ Learned positional embeddings
∘ Causal masking using lower-triangular matrix
∘ Scaled dot-product attention
∘ Dropout regularization

2️⃣ Custom Layer Normalization

∘ Manual implementation (gamma, beta parameters)
∘ Stabilizes training

3️⃣ Feed Forward Network

∘ Linear → ReLU → Linear
∘ Expansion factor: 2× embedding size
∘ Dropout applied

4️⃣ Final Linear Projection

∘ Maps embedding space → vocabulary logits

📈 Training Results

Training and validation loss progression:

Epoch :0  | loss : 6.4241 | val loss : 5.8490
Epoch :1  | loss : 5.6358 | val loss : 5.5810
Epoch :2  | loss : 5.3720 | val loss : 5.4729
Epoch :3  | loss : 5.2238 | val loss : 5.4107
Epoch :4  | loss : 5.1233 | val loss : 5.3685
Epoch :5  | loss : 5.0478 | val loss : 5.3337
Epoch :6  | loss : 4.9877 | val loss : 5.3084
Epoch :7  | loss : 4.9376 | val loss : 5.2866
Epoch :8  | loss : 4.8951 | val loss : 5.2693
Epoch :9  | loss : 4.8581 | val loss : 5.2552
Epoch :10 | loss : 4.8256 | val loss : 5.2448
Epoch :11 | loss : 4.7970 | val loss : 5.2342
Epoch :12 | loss : 4.7715 | val loss : 5.2237
Epoch :13 | loss : 4.7480 | val loss : 5.2199
Epoch :14 | loss : 4.7266 | val loss : 5.2108
Epoch :15 | loss : 4.7072 | val loss : 5.2058
Epoch :16 | loss : 4.6893 | val loss : 5.1992
Epoch :17 | loss : 4.6728 | val loss : 5.1957
Epoch :18 | loss : 4.6574 | val loss : 5.1924
Epoch :19 | loss : 4.6434 | val loss : 5.1913

∘ Final Validation Loss: 5.19

For a vocabulary of 16,000 tokens and 2.2M training tokens, this indicates strong learning and stable convergence.

🧪 Training Pipeline

∘ Load WikiText-2 dataset
∘ Clean and normalize text
∘ Train SentencePiece BPE tokenizer (vocab = 16,000)
∘ Encode full dataset into token IDs
∘ Create sliding window dataset:
∘ Input: tokens[i : i + block_size]
∘ Target: tokens[i+1 : i+block_size+1]
∘ Train with CrossEntropyLoss
∘ Save best model based on validation loss

🗂 Project Structure
Mini_gpt/
│
├── model.py                # Transformer architecture + training loop
├── model_dataset.py        # Sliding window dataset class
├── preprocessing.py        # Dataset cleaning + tokenization
├── prediction.py           # Text generation script
├── tokenizer_files/        # SentencePiece tokenizer
├── saved_tokens/           # Preprocessed token tensors
├── saved_model_and_files/  # Trained model weights



🔮 Text Generation (Prediction Pipeline)

∘ This project includes a fully functional autoregressive text generation script implemented in prediction.py.
∘ The generation pipeline loads the trained GPT-style decoder model and produces text token-by-token using probabilistic sampling.

⚙️ Generation Setup

∘ Model: Trained Decoder-only Transformer
∘ Embedding Dimension: 64
∘ Heads: 4
∘ Block Size: 16
∘ Tokenizer: SentencePiece (vocab = 16,000)
∘ Device: CUDA (if available) else CPU
∘ Default Max Length: 50 tokens
∘ The model weights are loaded from: Mini_gpt/saved_model_and_files/trained_model_full_dataset.pth

🧠 Generation Algorithm

The generation function follows standard autoregressive decoding:

1️⃣ Encode input text using SentencePiece
2️⃣ Convert tokens to tensor and move to device
3️⃣ Loop until maxlength is reached:

∘ Take last block_size tokens
∘ Forward pass through model
∘ Extract logits for last token
∘ Apply softmax to get probability distribution
∘ Sample next token using torch.multinomial
∘ Append token to sequence

4️⃣ Decode full token sequence back to text

🧪 Decoding Strategy

∘ Current decoding method: Softmax → Multinomial Sampling

∘ This allows stochastic text generation instead of deterministic greedy decoding.

∘ The script also includes a commented greedy alternative: torch.argmax(...)

📌 Example Prompt

∘ text = " what is the  "
∘ maxlength = 50

∘ output :-
what is the American Civil Service Department of Asian Studies , has been @-@ based on the two world can meet and bull in the 1906 Summer Olympics . The statistics is a tropical depression moved across the Pacific and Northern Ireland . The Autumn areas and caused by the Pope ..

∘ The model will continue generating text for up to 50 tokens.
∘ The symbols @-@ are denoting a hyphen(-) ; Another Example , if there is @+@ , it denotes the addition symbol(+).

🚀 How to Generate Text

Run:

python prediction.py

The script will:

∘Load trained model
∘ Generate text from prompt
∘ Print final output

🔍 Important Notes

∘ Context window is limited to block_size = 16
∘ Model always conditions on the last 16 tokens
∘ Uses autoregressive causal masking
∘ Sampling randomness may produce different outputs per run


🧠 Key Learnings

∘ Implemented masked self-attention from scratch
∘ Built custom LayerNorm instead of using nn.LayerNorm
∘ Understood causal masking and token shifting
∘ Applied AdamW with weight decay for Transformer training
∘ Built complete tokenizer-to-inference pipeline
∘ Diagnosed and resolved overfitting issues
∘ Improved generation quality via decoding strategy research
