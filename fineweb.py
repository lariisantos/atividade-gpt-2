import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configurações do Sharding de Dados
# -----------------------------------------------------------------------------
local_dir = "edu_fineweb10B"
shard_size = 100000  # Reduzido para 100k tokens por shard para ser leve na sua máquina

# Cria a pasta de destino
os.makedirs(local_dir, exist_ok=True)

# 1. Carrega uma amostra do dataset FineWeb-Edu da Hugging Face
print("Baixando amostra do dataset FineWeb-Edu...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # Token de fim de texto

def tokenize(doc):
    # Tokeniza um único documento e adiciona o token <|endoftext|> no final
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# -----------------------------------------------------------------------------
# Processamento e Salvamento dos Shards Binários
# -----------------------------------------------------------------------------
shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

print("Iniciando tokenização e criação dos arquivos binários (.npy/.bin)...")

# Processa os primeiros documentos do dataset
for doc in dataset:
    tokens = tokenize(doc)
    
    # Se o documento couber no shard atual
    if token_count + len(tokens) < shard_size:
        all_tokens_np[token_count:token_count + len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        # Preenche o restante do shard atual
        remainder = shard_size - token_count
        all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
        
        # Salva o shard completo em disco
        filename = os.path.join(local_dir, f"edufineweb_train_{shard_index:06d}.npy")
        write_datafile(filename, all_tokens_np)
        
        shard_index += 1
        progress_bar.close()
        progress_bar = None
        
        # Inicia o próximo shard com os tokens restantes
        tokens = tokens[remainder:]
        all_tokens_np[:len(tokens)] = tokens
        token_count = len(tokens)

    # Para após criar 2 shards de teste para não demorar
    if shard_index >= 2:
        break

print(f"\nSucesso! {shard_index} shards de dados foram criados na pasta '{local_dir}'.")