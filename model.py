import torch
import torch.nn as nn

class TransformerXL(nn.Module):
    def __init__(self, 
                 vocab_size,
                 time_vocab_size,
                 embed_dim=512,
                 seq_length=384,
                 batch_size=8,
                 n_layers=8,
                 n_heads=8,
                 d_ff=2048,
                 dropout=0.1):
        super(TransformerXL, self).__init__()
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.time_embedding = nn.Embedding(time_vocab_size, embed_dim)
        
        # Create encoder layer with updated dimensions
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        # Create transformer with specified number of layers
        self.transformer = nn.TransformerEncoder(
            encoder_layers,
            num_layers=n_layers
        )
        
        # Output layer
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # Combine token and positional embeddings
        x = self.embedding(x) + self.time_embedding(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Project to vocabulary size
        return self.fc(x)
    
    def generate(self, input_tensor, max_tokens=500):
        self.eval()  # Set to evaluation mode
        
        output = input_tensor.clone()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Ensure we don't exceed maximum sequence length
                if output.size(1) >= self.seq_length:
                    output = output[:, -self.seq_length:]
                    
                logits = self.forward(output)[:, -1, :]  # Get last token logits
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                output = torch.cat([output, next_token], dim=1)
                
                if next_token.item() == 0:  # Stop if EOS token
                    break
                    
        return output.squeeze(0).cpu().numpy()