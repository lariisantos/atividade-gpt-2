import torch
import torch.nn.functional as F
import tiktoken
from train_gpt2 import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_example(model, enc, context, completions):
    """
    Avalia qual das alternativas (completions) é a mais provável para o contexto fornecido.
    """
    context_tokens = enc.encode(context)
    losses = []

    for completion in completions:
        completion_tokens = enc.encode(completion)
        tokens = context_tokens + completion_tokens
        
        x = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(x)
            # Calcula a loss apenas na secção da resposta (completion)
            completion_len = len(completion_tokens)
            logits_comp = logits[0, -completion_len:, :50257]
            targets_comp = y[0, -completion_len:]
            
            loss = F.cross_entropy(logits_comp, targets_comp)
            losses.append(loss.item())

    # A opção correta é aquela com menor loss (maior probabilidade)
    pred_index = losses.index(min(losses))
    return pred_index, losses

if __name__ == "__main__":
    print("A carregar modelo GPT-2 pré-treinado para teste HellaSwag...")
    model = GPT.from_pretrained("gpt2").to(device)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")

    # Exemplo no formato do benchmark HellaSwag
    context = "A man is sitting at a piano."
    options = [
        " He starts playing a beautiful melody.",
        " He eats a bowl of soup with a spoon.",
        " The piano starts flying into space.",
        " He drives a truck down the highway."
    ]

    pred, losses = evaluate_example(model, enc, context, options)
    
    print(f"\nContexto: '{context}'")
    for i, opt in enumerate(options):
        print(f" Option {i}: '{opt}' | Loss: {losses[i]:.4f}")

    print(f"\nOpção selecionada pelo modelo: {pred} -> '{options[pred]}'")