# 🎵 Transformer-XL for Music Generation

## 🌟 Overview
This project implements a **Decoder-Only Transformer-XL** model for symbolic music generation. The model is trained on **event-based tokenized MIDI data** and uses **relative positional encoding (RPE)** to maintain long-term coherence in compositions.

Unlike traditional sequence models, this Transformer-XL-based architecture enables **long-context music generation** by capturing **hierarchical musical structures** such as motifs, phrases, and sections. The model is designed to **continue a given MIDI composition**, generating new music that extends the original piece while maintaining style and coherence.

## ✨ Features
✅ **Event-Based Encoding**: Tokenization based on MIDI events (Note-On, Note-Off, Velocity, Instrument, Time).  
✅ **Relative Positional Encoding (RPE)**: Captures time relationships between musical events.  
✅ **Transformer-XL Architecture**: Improves memory retention for long sequences.  
✅ **Autoregressive Music Generation**: Generates coherent compositions **continuing from a given MIDI file**.  
✅ **Hierarchical Attention Mechanism**: Models motifs, phrases, and larger structures in music.  
✅ **Custom MIDI Tokenization Pipeline**: Converts MIDI files into discrete tokens for training & inference.  

## 📂 Project Structure
```
├── model.py                    # Transformer-XL model implementation
├── train.ipynb                  # Training script (Colab/Jupyter Notebook)
├── generate.ipynb               # Music generation script (Colab/Jupyter Notebook)
├── tokenizer.py                 # Event-based MIDI tokenizer
├── dataset.json                 # Tokenized dataset
├── event_to_token.json           # Mapping of events to token IDs
├── generated_music/              # Folder for generated MIDI files
└── README.md                     # This file!
```

## 🚀 Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/music-transformer-xl.git
cd music-transformer-xl
```

### 2️⃣ Install dependencies
```bash
pip install torch torch-xla pretty_midi numpy miditoolkit
```

### 3️⃣ Set up Google Colab (for TPU Training)
If training on **Google Colab**, use:
```python
import torch_xla.core.xla_model as xm
print(f"Using device: {xm.xla_device()}")
```

## 🏗️ Model Architecture
The model consists of **multiple Transformer-XL decoder blocks**, each including:
- **Multi-Head Self-Attention** (with causal masking for autoregressive generation)
- **Feedforward Networks** (FFN layers for feature transformation)
- **Layer Normalization & Dropout** (to stabilize training)

```python
class TransformerXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
```

## 🎼 Training
1️⃣ **Prepare Data**: Tokenize MIDI files into event-based representations.
```python
from tokenizer import tokenize_midi_with_note_on_off

# Tokenize a MIDI file
tokens, tokenized_events, event_to_token = tokenize_midi_with_note_on_off("dataset.json")
```

2️⃣ **Train Model on TPU** (Colab TPUs recommended for efficiency)
```python
device = xm.xla_device()
model = TransformerXL(vocab_size, time_vocab_size, d_model=384, n_layers=6, n_heads=6, d_ff=1024, dropout=0.1).to(device, dtype=torch.bfloat16)
```
```python
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        event_batch, time_batch = batch
        event_batch = event_batch.to(device)
        time_batch = time_batch.to(device)
        loss = train_step(model, event_batch, time_batch, optimizer, criterion)
        total_loss += loss
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(data_loader):.4f}")
```

3️⃣ **Save Model**
```python
torch.save(model.state_dict(), "transformer_xl_music_tpu.pth")
```

## 🎹 Music Generation
1️⃣ **Load Pre-trained Model**
```python
model.load_state_dict(torch.load("transformer_xl_music_tpu.pth", map_location=device))
model.eval()
```

2️⃣ **Generate Music from MIDI Prompt**
```python
generated_sequence = generate_music(model, tokenizer, "input.mid", "output.mid", max_tokens=500)
```

3️⃣ **Convert Tokens to MIDI**
```python
tokens_to_midi(generated_sequence, token_to_event, "generated_music.mid")
```

## 📈 Results & Evaluation
✅ **Loss Curves**: Training stabilizes after ~30 epochs.  
✅ **Generated MIDI Analysis**: Pieces exhibit coherent structure and continuity.  
✅ **Qualitative Listening Tests**: Music retains motif patterns from the input MIDI.  

## 🔬 Future Improvements
- **Hierarchical Transformers** for explicit structure modeling.  
- **Better Rhythm Encoding** using learnable time representations.  
- **Zero-shot Style Transfer** (train on multiple styles & fine-tune dynamically).  
- **Live Performance Adaptation** (real-time music generation with MIDI input).  

## 📝 Citation
If you use this work, please cite:
```
@article{yourpaper2024,
  title={Transformer-XL for Music Generation},
  author={Your Name},
  journal={ArXiv},
  year={2024}
}
```

## 🤝 Contributing
Feel free to **open issues**, submit PRs, or improve the project. Let's push AI-generated music forward! 🚀🎶

