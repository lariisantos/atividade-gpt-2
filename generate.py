import torch
import torch.nn.functional as F
import tiktoken
from train_gpt2 import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carrega o GPT-2 oficial da OpenAI dentro da SUA classe GPT
print("Carregando pesos pré-treinados do GPT-2...")
model = GPT.from_pretrained("gpt2")
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")

def gerar_texto(prompt, max_tokens=50, top_k=50):
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :50257]
            probs = F.softmax(logits, dim=-1)
            
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
            
    return enc.decode(x[0].tolist())

prompt = "In a shocking finding, scientist discovered"
print(f"\n=== Prompt: '{prompt}' ===")
print(gerar_texto(prompt, max_tokens=50))