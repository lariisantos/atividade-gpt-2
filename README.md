# atividade-gpt-2
# Reprodução do GPT-2 (124M) em PyTorch

Este repositório contém a implementação completa e funcional da reconstrução do GPT-2 (124M) do zero utilizando PyTorch, desenvolvida com base no tutorial "Let's reproduce GPT-2" de Andrej Karpathy.

O projeto abrange desde a definição da arquitetura da rede neural até a integração de otimizações de nível de produção (FlashAttention, arredondamento de vocabulário, acúmulo de gradientes e suporte a treinamento distribuído), além da preparação de datasets binários e avaliação de benchmarks.

## Estrutura do Repositório

- **train_gpt2.py**: Implementação central da arquitetura GPT-2, rotina de inicialização personalizada de pesos, otimizador AdamW com weight decay seletivo, Cosine LR Scheduler, acúmulo de gradientes, suporte a DDP (Distributed Data Parallel) e loop de treino/validação.
- **fineweb.py**: Script para pré-processamento e tokenização de documentos em massa do dataset FineWeb-Edu, salvando os dados em shards binários (.npy) para leitura rápida.
- **generate.py**: Pipeline de inferência e amostragem top-k para geração de texto a partir de checkpoints salvos ou dos pesos pré-treinados da OpenAI.
- **hellaswag.py**: Pipeline de avaliação de acurácia no benchmark HellaSwag com mascaramento de perda focado exclusivamente nos tokens de resposta.
- **play.ipynb**: Notebook interativo de exploração para validação cruzada de logits entre a implementação própria e o modelo de referência da Hugging Face.

## Requisitos e Instalação

Recomenda-se o uso de ambiente virtual com Python 3.10+.

1. Clone o repositório:
   git clone https://github.com/lariisantos/atividade-gpt-2
   cd atividade-gpt-2

2. Instale as dependências do projeto:
   pip install torch tiktoken datasets transformers tqdm numpy

## Como Executar

### 1. Preparação dos Dados (FineWeb-Edu)
Para tokenizar os textos e gerar os arquivos binários de treinamento:
python fineweb.py

### 2. Treinamento do Modelo
Para executar o loop de treinamento:
python train_gpt2.py

Caso queira executar em cluster com múltiplas GPUs via DDP:
torchrun --standalone --nproc_per_node=2 train_gpt2.py

### 3. Geração de Texto (Inferência)
Para gerar continuações de texto com o modelo:
python generate.py

### 4. Avaliação no Benchmark HellaSwag
Para calcular a taxa de acerto do modelo no HellaSwag:
python hellaswag.py

## Otimizações de Engenharia Implementadas

- **FlashAttention (`F.scaled_dot_product_attention`)**: Cálculo eficiente da atenção causal sem alocação explícita da matriz de atenção em memória.
- **Vocabulário Arredondado (50.304 tokens)**: Redimensionamento das matrizes para múltiplos de 64, aumentando a eficiência de processamento em hardware (Tensor Cores).
- **Compartilhamento de Pesos (Weight Sharing)**: Associação direta dos pesos de embedding de entrada (`wte`) com a camada linear de saída (`lm_head`).
- **Acúmulo de Gradientes**: Divisão de lotes grandes em micro-etapas para simulação de batch sizes maiores em ambientes com limitação de memória.
- **Agendador Cosine LR com Warmup**: Aquecimento linear seguido de decaimento suave por cosseno até 10% da taxa máxima.
- **Distributed Data Parallel (DDP)**: Sincronização explícita de gradientes e redução de perdas entre múltiplos processos.