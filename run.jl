struct Config
    dim::Int32        # transformer dimension
    hidden_dim::Int32 # for ffn layers
    n_layers::Int32   # number of layers
    n_heads::Int32    # number of query heads
    n_kv_heads::Int32 # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int32 # vocabulary size, usually 256 (byte-level)
    seq_len::Int32    # max sequence length
end

struct TransformerWeights
    # token embedding table
    token_embedding_table::Array{Float32,2}    # (vocab_size, dim)
    # weights for rmsnorms
    rms_att_weight::Array{Float32,2} # (layer, dim) rmsnorm weights
    rms_ffn_weight::Array{Float32,2} # (layer, dim)
    # weights for matmuls
    wq::Array{Float32,3} # (layer, dim, dim)
    wk::Array{Float32,3} # (layer, dim, dim)
    wv::Array{Float32,3} # (layer, dim, dim)
    wo::Array{Float32,3} # (layer, dim, dim)
    # weights for ffn
    w1::Array{Float32,3} # (layer, hidden_dim, dim)
    w2::Array{Float32,3} # (layer, dim, hidden_dim)
    w3::Array{Float32,3} # (layer, hidden_dim, dim)
    # final rmsnorm
    rms_final_weight::Array{Float32,1} # (dim,)
    # freq_cis for RoPE relatively positional embeddings
    freq_cis_real::Array{Float32,2} # (seq_len, dim/2)
    freq_cis_imag::Array{Float32,2} # (seq_len, dim/2)
end

struct RunState
    # current wave of activations
    x # activation at current time stamp (dim,)
    xb # same, but inside a residual branch (dim,)
    xb2 # an additional buffer just for convenience (dim,)
    hb # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2 # buffer for hidden dimension in the ffn (hidden_dim,)
    q # query (dim,)
    k # key (dim,)
    v # value (dim,)
    att # buffer for scores/attention values (seq_len,)
    logits # output logits
    # kv cache
    key_cache   # (layer, seq_len, dim)
    value_cache # (layer, seq_len, dim)
end

checkpoint = "llama2.jl/out/model.bin"

## Define Config Header
f = open(checkpoint, "r")
config_vector = zeros(Int32, 7)
read!(f, config_vector)
config = Config(config_vector...)

## Define Transformer Weights
token_embedding_table = zeros(Float32, config.vocab_size, config.dim)
rms_att_weight = zeros(Float32, config.n_layers, config.dim)
wq = zeros(Float32, config.n_layers, config.dim, config.dim)
wk = zeros(Float32, config.n_layers, config.dim, config.dim)
wv = zeros(Float32, config.n_layers, config.dim, config.dim)
wo = zeros(Float32, config.n_layers, config.dim, config.dim)
rms_ffn_weight = zeros(Float32, config.n_layers, config.dim)
w1 = zeros(Float32, config.n_layers, config.dim, config.hidden_dim)
w2 = zeros(Float32, config.n_layers, config.hidden_dim, config.dim)
w3 = zeros(Float32, config.n_layers, config.dim, config.hidden_dim)
rms_final_weight = zeros(Float32, config.dim)
head_size::Int32 = config.dim / config.n_heads;
freq_cis_real = zeros(Float32, config.seq_len, trunc(Int, head_size / 2))
freq_cis_imag = zeros(Float32, config.seq_len, trunc(Int, head_size / 2))

read!(f, token_embedding_table)
read!(f, rms_att_weight)
read!(f, wq)
read!(f, wk)
read!(f, wv)
read!(f, wo)
read!(f, rms_ffn_weight)
read!(f, w1)
read!(f, w2)
read!(f, w3)
read!(f, rms_final_weight)
read!(f, freq_cis_real)
read!(f, freq_cis_imag)

weights = TransformerWeights(
    token_embedding_table,
    rms_att_weight,
    rms_ffn_weight,
    wq,
    wk,
    wv,
    wo,
    w1,
    w2,
    w3,
    rms_final_weight,
    freq_cis_real,
    freq_cis_imag
)

