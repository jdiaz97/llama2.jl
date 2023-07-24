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
    x::Array{Float32,1} # activation at current time stamp (dim,)
    xb::Array{Float32,1} # same, but inside a residual branch (dim,)
    xb2::Array{Float32,1} # an additional buffer just for convenience (dim,)
    hb::Array{Float32,1} # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Array{Float32,1} # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Array{Float32,1} # query (dim,)
    k::Array{Float32,1} # key (dim,)
    v::Array{Float32,1} # value (dim,)
    att::Array{Float32,1} # buffer for scores/attention values (seq_len,)
    logits::Array{Float32,1} # output logits
    # kv cache
    key_cache::Array{Float32,3}   # (layer, seq_len, dim)
    value_cache::Array{Float32,3} # (layer, seq_len, dim)
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

x = zeros(Float32, config.dim)
xb = zeros(Float32, config.dim)
xb2 = zeros(Float32, config.dim)
hb = zeros(Float32, config.hidden_dim)
hb2 = zeros(Float32, config.hidden_dim)
q = zeros(Float32, config.dim)
k = zeros(Float32, config.dim)
v = zeros(Float32, config.dim)
att = zeros(Float32, config.seq_len)
logits = zeros(Float32, config.vocab_size)
key_cache = zeros(Float32, config.n_layers, config.seq_len, config.dim)
value_cache = zeros(Float32, config.n_layers, config.seq_len, config.dim)

state = RunState(x, xb, xb2, hb, hb2, q, k, v, att, logits, key_cache, value_cache)

function copy!(a::Vector{Float32}, b::Vector{Float32}, size::Int32)
    for i in 1:size
        a[i] = b[i]
    end
end

function accum!(a::Vector{Float32}, b::Vector{Float32}, size::Int)
    for i in 1:size
        a[i] += b[i]
    end
end

function rmsnorm!(o, x, weight, size)
    # calculate sum of squares
    ss = 0.0
    for j in 1:size
        ss += x[j]^2
    end
    ss /= size
    ss += 1e-5
    ss = 1.0 / sqrt(ss)
    # normalize and scale
    for j in 1:size
        o[j] = weight[j] * (ss * x[j])
    end
end

function softmax!(x::Vector{Float32}, size::Int32)
    if size == 1 
        x[1] = 1.0f
    end
    # Find max value (for numerical stability)
    max_val = maximum(x)
    
    # E^x
    for i in 1:size
        x[i] = exp(x[i] - max_val)
    end
    # Normalize
    v_sum = sum(x)
    for i in 1:size
        x[i] /= v_sum
    end
end

function matmul!(xout, x, w)
    xout .= w * x
end

function transformer!(token::Int32, pos::Int32,p::Config, s::RunState, w::TransformerWeights)
    ## Convenience variables
    x::Vector{Float32} = s.x
    dim::Int32 = p.dim
    hidden_dim::Int32 = p.hidden_dim
    head_size::Int32 = trunc(Int32, dim/p.n_heads)

    ## copy the token embedding into x
    content_row = weights.token_embedding_table[token,:]
    copy!(x, content_row, dim)

    freq_cis_real_row = w.freq_cis_real[trunc(Int,pos * head_size / 2),:]
    freq_cis_imag_row = w.freq_cis_imag[trunc(Int,pos * head_size / 2),:]
    
    for l in 1:config.n_layers
        ##  attention rmsnorm
        rmsnorm!(s.xb, s.x, w.rms_att_weight[l,:], dim);

        # qkv matmuls for this position
        matmul!(s.q, s.xb, w.wq[l,:,:])
        matmul!(s.k, s.xb, w.wk[l,:,:])
        matmul!(s.v, s.xb, w.wv[l,:,:])

        ## apply RoPE rotation to the q and k vectors for each head

        for h in 1:config.n_heads
            ## get the q and k vectors for this head
            q = s.q
            k = s.k
            
            for i in 1:2:head_size
                println(i)
                q0 = q[i]
                q1 = q[i+1]
                k0 = k[i]
                k1 = k[i+1]
                fcr = freq_cis_real_row[trunc(Int,i/2+0.5)];
                fci = freq_cis_imag_row[trunc(Int,i/2+0.5)];
                q[i]   = q0 * fcr - q1 * fci;
                q[i+1] = q0 * fci + q1 * fcr;
                k[i]   = k0 * fcr - k1 * fci;
                k[i+1] = k0 * fci + k1 * fcr;
            end
        end

        #  save key,value at this time step (pos) to our kv cache
        ## WHAT IS LOFF guys
        loff = l * p.seq_len * dim ##  kv cache layer offset for convenience
        key_cache_row = s.key_cache[loff[pos,dim]]
        value_cache_row = s.value_cache[loff[pos,dim]]
        copy!(key_cache_row, s.k, dim);
        copy!(value_cache_row, s.v, dim);
    end
    
end

transformer!(Int32(3),Int32(4),config,state,weights)



