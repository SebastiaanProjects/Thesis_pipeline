import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import datetime
import numpy as np

class TimeSeriesMAEEncoder(nn.Module):
    """
    A Transformer-based encoder for time series data. This encoder projects input
    data into a higher-dimensional space using a linear transformation followed
    by positional encoding and multiple layers of Transformer encoder layers. This
    allows the model to capture complex dependencies across time steps in the sequence.

    Args:
        segment_dim (int): Dimensionality of each segment in the input data.
        embed_dim (int): Dimensionality of the embedding space into which the input data is projected.
        num_heads (int): Number of attention heads in each Transformer encoder layer.
        num_layers (int): Number of Transformer encoder layers.
        dropout_rate (float): Dropout rate to use in each Transformer encoder layer for regularization.

    Attributes:
        linear_proj (torch.nn.Linear): Linear projection layer that transforms input data to the embedding dimension.
        positional_embeddings (torch.nn.Parameter): Learnable positional embeddings added to the linearly projected data to retain positional information.
        transformer_encoder (torch.nn.TransformerEncoder): The Transformer encoder stack composed of multiple layers defined by `num_layers`.

    Methods:
        forward(x, mask):
            Processes the input `x` through the encoder stack applying linear projection,
            adding positional embeddings, and then passing through multiple Transformer
            encoder layers. The `mask` parameter is intended for use with masked attention
            mechanisms if required but is not utilized directly in this simple implementation.
            
            Args:
                x (torch.Tensor): Input data tensor of shape [batch_size, seq_len, segment_dim].
                mask (torch.Tensor): A boolean tensor of shape [batch_size, seq_len] indicating
                                     where to apply masking, though it is not used in this implementation.

            Returns:
                torch.Tensor: The encoded output from the Transformer encoder stack, preserving
                              the sequence length but with features transformed into the embedding dimension.
    """
    def __init__(self, segment_dim, embed_dim, num_heads, num_layers, dropout_rate,sequence_length=200): #dropout_rate is for robustness, different than masked encoding done here
        super(TimeSeriesMAEEncoder, self).__init__() #segment_dimension is the amount of values in one datapoint, three in this case
        #embed_dimension is the hyperparameter stating how complex the data can be scoped out 
        #num_heads is the amount of different parts of the input that can be attended to, to caputure a richer set of dependencies. The embed_dimensions should be divisible by num_heads. 
        #num_layers: each layer consists of a multi-head attention mechanism followed by a feed-forward neural network, and having multiple layers allows the model to learn complex representations.More is not always better due to overfitting. Just do four or less
        self.segment_dim = segment_dim # Dimension of each time-series segment
        self.embed_dim = embed_dim  # Dimension of the embedding space
        
        self.linear_proj = nn.Linear(segment_dim, embed_dim)#.to(device) #in a linear way changes the 4value segment dimension into the chosen embedded dimension increasing complexity of the data
        #when the data for the positional_embeddings are done for the seaguls change the nn.Parameter(torch.randn(1,20,embed_dim))
        #self.positional_embeddings = nn.Parameter(torch.randn(1, 200, embed_dim))#.to(device) #creating positional embedding as a training parameter for the nerual network which is important for the effectiveness of the transformer model in processing sequences where the order of elements is important
        self.positional_embeddings = nn.Parameter(torch.randn(1,sequence_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate,batch_first=True)#.to(device) #initializing the encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)#.to(device)

    def forward(self, x, mask): #the mask is only given such that the decoder can work with the mask straight away
        # x is the input segments of shape [batch_size, seq_len, segment_dim]
        x = x.float()  # Make sure input data is float
        #x = x.to(device)
        #print(f"Input data statistics - min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}, std: {x.std().item()}")
        #print(f"Initial weights - min: {self.linear_proj.weight.min().item()}, max: {self.linear_proj.weight.max().item()}")
        #print(f"Initial biases - min: {self.linear_proj.bias.min().item()}, max: {self.linear_proj.bias.max().item()}")
        if torch.isnan(x).any():
            print("NaN detected in input data")

        bias = self.linear_proj.bias.data
        weights = self.linear_proj.weight.data
        moment = datetime.datetime.now().strftime("%H-%M-%S")
        #np.savetxt(f"test_output/bias_{moment}", bias.detach().cpu().numpy())
        #np.savetxt(f"test_output/weight_{moment}", weights.detach().cpu().numpy())

        x_projected = self.linear_proj(x) #apply the earlier created lineair_projection

        if torch.isnan(x_projected).any():
            print("NaN detected in linear projection")
        bias = self.linear_proj.bias.data
        weights = self.linear_proj.weight.data

        # Create positional embeddings for each data point in the batch
        # (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        positional_embeddings = self.positional_embeddings[:, :seq_len, :]
        if torch.isnan(positional_embeddings).any():
            print("NaN detected in positional_embeddings")

        # Add positional embeddings to the linearly projected data
        x_projected += positional_embeddings

        # Apply Transformer encoder to the projected data with positional embeddings
        # Note that the mask is not directly used here - this example assumes you have already used the mask to filter your data
        encoder_output = self.transformer_encoder(x_projected)

        return encoder_output    

class TimeSeriesMAEDecoder(nn.Module):
    def __init__(self, embed_dim, decoder_embed_dim, num_heads, num_layers, max_seq_length, dropout_rate):
        """
        Initializes a Transformer-based decoder module for use in a Masked Autoencoder framework, specifically designed
        for time series data reconstruction. The decoder reconstructs the original data from encoded patches that were
        partially masked during the encoding process.

        Args:
            embed_dim (int): The dimensionality of the embeddings for the original input data.
            decoder_embed_dim (int): The internal embedding dimension used by the decoder.
            num_heads (int): The number of attention heads in the Transformer decoder.
            num_layers (int): The number of Transformer layers in the decoder.
            max_seq_length (int): The maximum length of the sequences that can be processed by the decoder.
            dropout_rate (float): Dropout rate used in the Transformer decoder for regularization.

        Attributes:
            mask_tokens (torch.nn.Parameter): Learnable mask tokens that are used to replace masked elements of the input during decoding.
            positional_embeddings (torch.nn.Parameter): Learnable positional embeddings added to the inputs for retaining positional information.
            transformer_decoder (torch.nn.TransformerDecoder): The Transformer decoder stack.
            output_proj (torch.nn.Linear): A linear layer to project the decoder's output back to the original embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim  # Target embedding dimension
        self.decoder_embed_dim = decoder_embed_dim  # Decoder's internal embedding dimension
        self.num_heads = num_heads
        self.num_layers = num_layers            
        self.max_seq_length = max_seq_length  # Reflects the maximum sequence length in your data
        
        # Initialize learnable mask tokens and positional embeddings
        # Mask tokens should be a 3D tensor: 1 x 1 x decoder_embed_dim
        self.mask_tokens = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))#.to(device)
        
        # Positional embeddings for the decoder
        self.positional_embeddings = nn.Parameter(torch.randn(1, max_seq_length, decoder_embed_dim))#.to(device) #make positional embedding for the 
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_embed_dim, nhead=num_heads, dropout=dropout_rate)#.to(device) #initialize the decoderlayers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)#.to(device)
        
        # Output projection layer to match the original data dimensionality
        self.output_proj = nn.Linear(decoder_embed_dim, embed_dim)#.to(device)

    def forward(self, encoded_patches, binary_mask):
        """
        Forward pass through the decoder.

        Args:
            encoded_patches (torch.Tensor): The encoded input data with dimensions (batch_size, seq_len, embed_dim), where missing patches are masked.
            binary_mask (torch.Tensor): A binary mask indicating which patches were masked, with dimensions (batch_size, seq_len, 1).

        Returns:
            torch.Tensor: The reconstructed output with the same dimensions as `encoded_patches`, attempting to reconstruct the original input data.
        """
        batch_size, seq_len, _ = encoded_patches.size()
        
        decoder_input = torch.where(binary_mask, encoded_patches, self.mask_tokens.expand(batch_size, seq_len, self.decoder_embed_dim))#, device=device)
        
        decoder_input += self.positional_embeddings[:, :seq_len, :]
        
        decoded_output = self.transformer_decoder(decoder_input, memory=encoded_patches)
        decoded_output = self.output_proj(decoded_output)
        
        return decoded_output


def clip_and_threshold_gradients(parameters, clip_value=1.0, min_value=1e-10):
    """
    Clips gradients by their norm and thresholds gradients to avoid very small values.
    
    Args:
        parameters (Iterable[torch.Tensor]): Model parameters.
        clip_value (float): Maximum allowed norm for gradients.
        min_value (float): Minimum allowed value for gradient magnitudes.
    """
    for param in parameters:
        if param.grad is not None:
            # Clip gradients by their norm
            nn_utils.clip_grad_norm_([param], max_norm=clip_value)
            # Ensure gradient magnitudes are above the min_value
            grad_data = param.grad.data
            small_grad_mask = torch.abs(grad_data) < min_value
            grad_data[small_grad_mask] = torch.sign(grad_data[small_grad_mask]) * min_value
            param.grad.data = grad_data

class ClipConstraint():
    """
    Clips tensor values
    """
    def __init__(self, clip_value: float) -> None:
        self.clip_value = clip_value

    def __call__(self, model) -> torch.Any:
        for p in model.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)