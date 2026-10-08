# 🧠 Dostoevsky-GPT

> A small GPT-style language model trained from scratch and fine-tuned on a corpus of works by Fyodor Dostoevsky.

Dostoevsky-GPT is an experimental autoregressive language model built to understand the fundamentals of modern language-model training from the ground up.

The model was first pretrained on the **TinyStories** dataset and subsequently fine-tuned on a curated **Dostoevsky text corpus** to learn literary vocabulary, sentence structures, dialogue patterns, and narrative style.

---

## ✨ Highlights

- 🏗️ GPT-style Transformer built and trained from scratch
- 📚 Fine-tuned on an ~8.4 MB Dostoevsky text corpus
- 🔤 GPT-2 tokenizer
- 🎯 Autoregressive next-token prediction
- 📈 Training and validation loss tracked throughout training
- 🧪 Evaluated using validation loss, perplexity, and top-1 next-token accuracy
- ✍️ Generates Dostoevsky-inspired literary continuations
- 🤗 Dataset and model checkpoint publicly available on Hugging Face

---

## 🏗️ Model Pipeline

```text
                 TinyStories
                     │
                     ▼
          ┌─────────────────────┐
          │   GPT Pretraining   │
          │                     │
          │  Transformer LM     │
          └──────────┬──────────┘
                     │
                     ▼
              GPT Checkpoint
                     │
                     ▼
          ┌─────────────────────┐
          │ Dostoevsky Corpus   │
          │       ~8.4 MB       │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Fine-Tuning         │
          │                     │
          │ Next-token          │
          │ prediction          │
          └──────────┬──────────┘
                     │
                     ▼
              Dostoevsky-GPT
                     │
                     ▼
              Text Generation
```

---

# 📊 Training Results

The model was evaluated on a held-out validation set during training.

| Metric | Result |
|---|---:|
| Validation Loss | **4.109** |
| Top-1 Next-Token Accuracy | **28.72%** |
| Validation Perplexity | **~60.9** |
| Training Objective | **Causal Language Modeling** |

### What does 28.72% mean?

The model's **top-1 next-token prediction** was correct approximately **28.72% of the time** on the validation set.

This measures exact next-token prediction and should not be interpreted as general language-understanding accuracy.

---

# 📈 Training vs Validation Loss

The following graph shows the training and validation loss throughout training.


Then it will appear here:
<img width="575" height="455" alt="image" src="https://github.com/user-attachments/assets/a802c608-dc66-4c08-821c-cf8ab6401962" />

### Interpreting the graph

A decreasing training loss indicates that the model is improving on the training data.

A decreasing validation loss indicates that the model is also improving on unseen data.

Comparing both curves helps identify potential **overfitting** and understand how well the model generalizes.

---

# ✍️ Sample Generation

## Prompt 1

```text
The sky was
```

### Generated

```text
The sky was gloomy and silent. But he had to be seen, and the same 
the morning he went to the window, and turned was in, and looked at 
him with a smile on his face. His face turned with a frownkin and 
whhewly, though he was watching, he was standing at the table.
```

---

## Prompt 2

```text
It was a dark and cold night
```

### Generated

```text
It was a dark and cold night. It was not a night to look. 
“I have been a very cold day,” said Shatov, “I’m not afraid of it. 
You don’t know it, of a dream,” Mitya muttered, waving his hand.
```

---

## Prompt 3

```text
The old man walked slowly
```

### Generated

```text
The old man walked slowly, as if he could, he had not seen some 
his time to say, and he did not know what he was doing, but his 
father in the story, he began to explain the word of the question.
```

> The generated samples demonstrate that the model has learned recognizable literary vocabulary, dialogue structures, narrative patterns, and character names. However, long-range coherence remains a limitation of the relatively small model.

---

# 🧮 Training Objective

The model uses **causal language modeling**.

Given a sequence of tokens:

```text
The old man walked
```

the model learns to predict the next token at every position:

```text
The  → old
old  → man
man  → walked
walked → ...
```

The training objective is cross-entropy loss:

\[
\mathcal{L}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log P(x_t|x_{<t})
\]

In simple terms, the model learns:

```text
P(next token | previous tokens)
```

---

# 🔬 Evaluation

## Validation Loss

Validation loss measures how well the model predicts tokens on data that was not used for updating the model's weights.

**Lower is better.**

## Top-1 Next-Token Accuracy

Measures how often the model's highest-probability prediction exactly matches the actual next token.

**Higher is better.**

## Perplexity

Perplexity is calculated from the validation loss:

\[
PPL=e^{Loss}
\]

For a validation loss of approximately `4.109`:

\[
e^{4.109}\approx60.9
\]

Therefore:

**Validation Perplexity ≈ 60.9**

Lower perplexity generally indicates lower uncertainty in the model's predictions.

---

# 🧱 Architecture

The model follows a **GPT-style decoder-only Transformer architecture**.

```text
Input Tokens
     │
     ▼
Token Embeddings
     +
Positional Information
     │
     ▼
┌─────────────────────┐
│  Transformer Block  │
│                     │
│  Masked Self-Attn   │
│          ↓          │
│  Feed Forward       │
│          ↓          │
│  Layer Normalization│
└──────────┬──────────┘
           │
           ▼
      Repeated N times
           │
           ▼
   Language Model Head
           │
           ▼
       Token Logits
           │
           ▼
     Next Token
```

---

# 🛠️ Tech Stack

- Python
- PyTorch
- tiktoken
- NumPy
- Matplotlib
- Google Colab
- Hugging Face

---

# 📚 Dataset

The model was fine-tuned using a cleaned text corpus containing works by **Fyodor Dostoevsky**.

The processed corpus is approximately **8.4 MB** in plain-text format.

### Hugging Face Dataset

**ShahirDevs/Dostoevsky**

The dataset repository contains the processed text corpus used for fine-tuning.

---

# 🤗 Model

The trained checkpoint is available on Hugging Face:

**ShahirDevs/Dostoevsky-Gpt**

The repository contains the trained model checkpoint and model card.

---

# 🚀 Usage

> The checkpoint uses a custom GPT architecture. The model architecture definition used during training is required to reconstruct the model from the `.pt` checkpoint.

## Install Dependencies

```bash
pip install torch tiktoken
```

## Load the Model

```python
import torch
import tiktoken

# GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load checkpoint
checkpoint = torch.load(
    "dostoevsky_gpt.pt",
    map_location="cpu"
)

# Recreate the model using the same
# architecture configuration used during training.

model = GPT(...)

model.load_state_dict(checkpoint)
model.eval()
```

## Generate Text

```python
prompt = "The sky was"

tokens = tokenizer.encode(prompt)

idx = torch.tensor([tokens])

with torch.no_grad():
    output = model.generate(
        idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40
    )

generated_text = tokenizer.decode(
    output[0].tolist()
)

print(generated_text)
```

> **Note:** The `GPT(...)` configuration must exactly match the architecture used to create the checkpoint.

---

# ⚠️ Limitations

This is a relatively small experimental language model.

The model may produce:

- Grammatically incorrect sentences
- Repetitive text
- Broken sentence structures
- Inconsistent characters
- Loss of context over longer generations
- Semantically incoherent passages

The model should therefore be considered an **educational and experimental language-modeling project**, rather than a production-ready language model.

Generated text is **Dostoevsky-inspired** and is not guaranteed to reproduce authentic Dostoevsky passages.

---

# 🎯 Why I Built This

The goal of this project was not simply to use an existing LLM.

I wanted to understand what actually happens **under the hood** when training a language model:

- Tokenization
- Embeddings
- Self-attention
- Transformer blocks
- Causal masking
- Cross-entropy loss
- Backpropagation
- Optimization
- Validation
- Perplexity
- Next-token prediction
- Fine-tuning
- Text generation

Building the model from scratch made these concepts much more concrete than simply calling an LLM API.

---

# 🔮 Future Improvements

- [ ] Increase model size
- [ ] Train on a larger literary corpus
- [ ] Improve long-range context retention
- [ ] Evaluate top-5 next-token accuracy
- [ ] Add automated perplexity evaluation
- [ ] Compare different decoding strategies
- [ ] Experiment with temperature and sampling
- [ ] Package the model for easier inference
- [ ] Deploy an interactive text-generation demo

---

# 📜 License

See the repository license for details.

---

# 👨‍💻 Author

**Shahir Ali**

Computer Science & Information Technology

Interested in **AI/ML, Deep Learning, Generative AI, and building models from first principles.**
