import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import FeedForward, LayerNorm

# ────────────────────────────────────────────────────────────────────────────────- #
# MULTI-HEAD SELF-ATTENTION MODULE                                                  #
#                                                                                   #
# This implements the key innovation of Transformer decoders:                       #
# causal (unidirectional) multi-head self-attention.                                #
#                                                                                   #
# Each token attends only to itself and prior tokens (never future tokens).         #
# The causal mask enforces this autoregressive constraint for next-token prediction #
#                                                                                   #
# Output is a set of context-aware token embeddings passed to residual connection.  #
# ────────────────────────────────────────────────────────────────────────────────- #

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, ctx_len, dropout, n_heads):
        """
        Multi-head self-attention layer with causal mask.

        Args:
            dim_in: input embedding dimension
            dim_out: output embedding dimension (must be divisible by n_heads)
            ctx_len: maximum sequence/context length
            dropout: dropout probability
            n_heads: number of attention heads
        """

        super().__init__()
        assert dim_out % n_heads == 0, "dim_out must be divisible by num_heads"

        self.dim_out = dim_out
        self.n_heads = n_heads
        self.head_dim = dim_out // n_heads
        
        # Linear projections for queries, keys, and values
        self.W_Q = nn.Linear(dim_in, dim_out, bias=False)
        self.W_K = nn.Linear(dim_in, dim_out, bias=False)
        self.W_V = nn.Linear(dim_in, dim_out, bias=False)

        # Output projection after attention
        self.out_project = nn.Linear(dim_out, dim_out)

        self.dropout = nn.Dropout(dropout)

        # Causal mask → upper triangle True where j > i (future positions blocked)
        # shape: (context_length, context_length)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool()
        )

    def forward(self, x):
        batch, n_tokens, _ = x.shape

        # Linear projections → shape [batch, seq_len, n_heads, head_dim]
        Q = self.W_Q(x).view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1,2)
        K = self.W_K(x).view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1,2)
        V = self.W_V(x).view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1,2)

        # Compute attention scores: shape [b, num_heads, seq_len, seq_len]
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask - broadcast across batch & heads
        causal_mask = self.mask[:n_tokens, :n_tokens]
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )

        # Softmax & dropout
        atten_weights = torch.softmax(scores, dim=-1)
        atten_weights = self.dropout(atten_weights)

        # Weighted sum of values
        context = atten_weights @ V

        # Rearrange back to [b, seq_len, d_out]
        context = context.transpose(1, 2).contiguous().view(batch, n_tokens, self.dim_out)

        return self.out_project(context)

# ──────────────────────────────────────────────────────────────────────────────── #
# DECODER BLOCK                                                                    #
#                                                                                  #
# Each decoder block follows the Transformer decoder design:                       #
# - masked multi-head self-attention (causal)                                      #
# - feedforward network                                                            #
# - layer normalization before each sub-layer                                      #
# - residual connection after each sub-layer                                       #
#                                                                                  #
# This forms a “stackable” unit for GPT’s decoder-only pipeline.                   #
# ──────────────────────────────────────────────────────────────────────────────── #

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.atten = MultiHeadAttention(
            dim_in=config["emb_dim"],
            dim_out=config["emb_dim"],
            ctx_len=config["context_length"],
            dropout=config["drop_rate"],
            n_heads=config["n_heads"]
        )
        self.feedfw = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # --- Self-Attention block ----------- #
        shortcircuit = x                # Store input for residual
        x = self.norm1(x)               # Pre-norm layer normalization
        x = self.atten(x)               # Apply masked multi-head attention
        x = self.drop_shortcut(x)       # Dropout after attention
        x = x + shortcircuit

        # FeedForward block
        shortcircuit = x
        x = self.norm2(x)               # Pre-norm before feedforward
        x = self.feedfw(x)              # Apply feedforward network
        x = self.drop_shortcut(x)
        x = x + shortcircuit

        return x

# ──────────────────────────────────────────────────────────────────────────────── #
# GPT MODEL                                                                        #
#                                                                                  #
# This is an autoregressive decoder-only Transformer (GPT-style).                  #
# It stacks N identical decoder blocks over token and position embeddings.         #
# The output is logits over vocabulary for next-token prediction.                  #
#                                                                                  #
# Key difference from encoder-decoder: no cross-attention, only masked self-attn.  #
# ──────────────────────────────────────────────────────────────────────────────── #

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout_emb = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, input_idx):
        batch_size, seq_len = input_idx.shape

        # Token embedding lookup → [batch, seq_len, emb_dim]
        token_embeds = self.token_emb(input_idx)
        
        # Positional embedding (same for all batches) → [seq_len, emb_dim]
        position_embeds = self.position_emb(torch.arange(seq_len, device=input_idx.device))

        # Sum token and positional embeddings
        x = token_embeds + position_embeds
        x = self.dropout_emb(x)

        # Pass forward through stacked decoder blocks
        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        # Final linear projection to vocab
        return self.out_head(x)
