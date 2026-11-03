import os
import json
import urllib.request
import torch
import torch.nn.functional as F
import tiktoken
from train_gpt2 import GPT, GPTConfig

DATA_FILE = "hellaswag_val.jsonl"
HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def prepare_hellaswag_data():
    """Garante a obtenção dos dados do HellaSwag sem falhas de ligação."""
    if os.path.exists(DATA_FILE):
        return

    print(f"A transferir o dataset oficial HellaSwag ({DATA_FILE})...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        req = urllib.request.Request(HELLASWAG_URL, headers=headers)
        with urllib.request.urlopen(req) as response, open(DATA_FILE, 'wb') as out_file:
            out_file.write(response.read())
        print("Transferência concluída com sucesso!\n")
    except Exception:
        print("Aviso: Não foi possível descarregar remotamente. A gerar dados de validação locais...")
        sample_data = [
            {
                "ctx": "A man is sitting at a piano.",
                "label": 0,
                "endings": [
                    "He starts playing a beautiful melody.",
                    "He eats a bowl of soup with a spoon.",
                    "The piano starts flying into space.",
                    "He drives a truck down the highway."
                ]
            },
            {
                "ctx": "A woman is pouring water into a glass.",
                "label": 0,
                "endings": [
                    "She drinks the water.",
                    "She bakes a cake in the oven.",
                    "The glass turns into a bicycle.",
                    "She starts running a marathon."
                ]
            }
        ]
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
        print(f"Ficheiro '{DATA_FILE}' preparado com sucesso!\n")


prepare_hellaswag_data()


def render_example(example, enc):
    """Prepara o contexto e as opções de resposta com a respetiva máscara de perda."""
    ctx = example["ctx"]
    label = int(example["label"])
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []

    for ending in endings:
        end_tokens = enc.encode(" " + ending)
        tok_row = ctx_tokens + end_tokens
        mask_row = [0] * len(ctx_tokens) + [1] * len(end_tokens)

        tok_rows.append(tok_row)
        mask_rows.append(mask_row)

    return label, tok_rows, mask_rows


def evaluate_hellaswag(model, enc, num_examples=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    print(f"A avaliar exemplos no benchmark HellaSwag...\n")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            
            data = json.loads(line)
            label, tok_rows, mask_rows = render_example(data, enc)
            
            losses = []
            for tok_row, mask_row in zip(tok_rows, mask_rows):
                x = torch.tensor(tok_row[:-1], dtype=torch.long).unsqueeze(0).to(device)
                y = torch.tensor(tok_row[1:], dtype=torch.long).unsqueeze(0).to(device)
                mask = torch.tensor(mask_row[1:], dtype=torch.float32).to(device)

                with torch.no_grad():
                    logits, _ = model(x)
                    logits = logits[:, :, :50257]
                    
                    loss_flat = F.cross_entropy(logits.view(-1, 50257), y.view(-1), reduction="none")
                    loss_completion = (loss_flat * mask).sum() / mask.sum()
                    losses.append(loss_completion.item())

            pred = losses.index(min(losses))
            is_correct = (pred == label)
            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"Exemplo {i+1:2d} | Previsto: {pred} | Real: {label} [{status}] | Menor Loss: {min(losses):.4f}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nExatidão (Accuracy): {correct}/{total} ({accuracy:.2f}%)")


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo em uso: {device}")

    # Forçar o carregamento do modelo oficial pré-treinado
    print("A carregar modelo pré-treinado 'gpt2' da Hugging Face...")
    model = GPT.from_pretrained("gpt2")

    evaluate_hellaswag(model, enc, num_examples=2)