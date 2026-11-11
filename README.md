# QWEN-30BRust — 0.2.1

Motor de inferência próprio para Qwen3-Coder-30B-A3B, em Rust e Vulkan/WGSL, com os pesos treinados originais em GGUF. A versão 0.2 mantém as ativações e o cache KV na GPU e executa os especialistas em lote. Não depende de Ollama, llama.cpp ou PyTorch para inferência.

**Estado:** experimental. A versão 0.2.1 acrescenta reciclagem segura de buffers de pesos e diagnóstico dos tempos de upload/espera.  A versão 0.1 foi executada pelo operador na RX 5700 XT e gerou texto. A versão 0.2 passou nos testes numéricos e no benchmark sintético com Vulkan por software; a velocidade do 30B real na Radeon precisa ser medida com `Benchmark.ps1`. Consulte [VALIDACAO.md](VALIDACAO.md) para resultados e limitações.

## Executar no Windows

Use Rust stable atualizado, MSVC/Windows SDK e o driver AMD com Vulkan. Na primeira instalação, Rust pode solicitar as ferramentas de C++ do Visual Studio Build Tools. Se Cargo não for encontrado, reabra o terminal; os scripts também tentam localizar `%USERPROFILE%\.cargo\bin`.

Clone a branch indicada no PR, ou `main` depois da integração. Abra o PowerShell na raiz do repositório:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
rustup update stable
rustup component add rustfmt clippy
.\Verificar.ps1 -GPU
.\Rodar.ps1 -Adapter '5700' -Prompt 'Crie uma funcao Rust para ordenar uma lista.'
```

A política de execução acima vale apenas para a janela atual. `Rodar.ps1` compila o executável, localiza o blob de `qwen3-coder:30b` no armazenamento local do Ollama e reutiliza seus pesos sem iniciar o Ollama. Respeita `OLLAMA_MODELS`; se necessário, informe `-Model 'D:\Modelos\modelo.gguf'`.

O script baixa apenas o tokenizer oficial, por revisão e SHA-256 fixados. Os pesos reais não estão neste repositório e não são baixados automaticamente. O tokenizer é conferido contra a tabela GGUF, e os tokens extras de padding dos embeddings são mascarados.

Para executar o caminho antigo como diagnóstico:

```powershell
.\Rodar.ps1 -Reference -Adapter '5700' -Prompt 'Explique ownership em Rust.'
```

## Medir no seu PC

```powershell
.\Benchmark.ps1 -Adapter '5700' -Context 4096 -GpuCacheMiB 4096 -Runs 3 -Warmup 1
```

O teste executa os modos de referência e residente, **um por vez**, com o mesmo arquivo, precisão FP32 e sequência explícita de tokens. Guarda `benchmark-result.json`, verifica todos os prefixos de logits e informa tokens/s, razão entre tempos, transferências, readbacks e alocações. O primeiro ciclo é reportado separado dos ciclos medidos. Um erro de paridade termina com código de falha; uma razão de velocidade não é garantia de correção.

Para isolar o efeito da reciclagem sem mudar o grafo residente ou a política do cache:

```powershell
.\Benchmark.ps1 -Adapter '5700' -CompareWeightReuse -OutFile 'benchmark-weight-reuse.json'
```

Esse modo serializa reciclagem desligada/ligada, valida os logits e grava `without_reuse`/`with_reuse`. A comparação padrão de grafo mantém as chaves `reference`/`resident`. `Rodar.ps1 -NoWeightReuse` ou `--no-weight-reuse` na CLI desativa a reciclagem para diagnóstico. Reduzir alocações não garante melhorar tokens/s; a versão 0.2.1 não altera o tamanho do cache nem diminui os bytes transferidos para os mesmos acessos.

É um teste de forward em prefixos fixos, não um benchmark de conversa natural nem de qualidade. Cache de páginas do sistema, ordem dos modos e estado do driver podem afetar o resultado; repita a medição. Para uma primeira comparação mais curta, use `-Runs 1 -Warmup 0`. O caminho de referência pode levar tempo.

`Rodar.ps1` inclui `decode_stats` separado de `prefill_stats`, bytes de pesos solicitados, alocações/reciclagens e tempos de parede de preparo de pesos e espera no readback. Esses tempos não são timestamps de kernels nem contadores de leitura de disco. Também reporta tempo até o primeiro token e `decode_forward_tokens_per_second`. Essa última métrica usa os forwards reais de geração e exclui amostragem/impressão; o primeiro token gerado usa os logits do último token do prompt e não conta como novo forward.

## Antes e depois

| Aspecto | Referência com execução da v0.1 | Residente 0.2 |
| --- | --- | --- |
| Ativações intermediárias | Retornam à CPU após cada matriz | Permanecem na GPU |
| RMSNorm, RoPE, atenção GQA e residuais | CPU | GPU |
| Cache KV FP32 | RAM | VRAM |
| Matrizes dos 8 especialistas por camada | 24 dispatches separados | 3 dispatches com dimensão de lote |
| Leituras síncronas para um token com logits no 30B de 48 camadas | 1393 | 49: 48 routers e logits finais |
| Envios explícitos por token com logits | 1393 | 49 |
| Buffers de trabalho | Recriados por matriz | Reutilizados entre tokens |
| Projeção do vocabulário durante prefill | Calculada por token | Calculada só no último token |
| Amostragem top-k | Ordenação de todo o vocabulário | Seleção parcial e ordenação do subconjunto |
| Descarte do cache de pesos | Busca linear da entrada mais antiga | Heap LRU com compactação de entradas antigas |

A leitura do router na CPU permanece para decidir quais especialistas carregar. Embeddings, roteamento, tokenização e amostragem continuam na CPU. O cache de pesos permanece separado dos bancos reutilizáveis dos especialistas. Não houve poda de camadas, redução de especialistas ativos nem troca dos pesos ou da arquitetura.

## Memória na RX 5700 XT

- O GGUF examinado tem aproximadamente 18,6 GB; a placa tem 8 GB de VRAM. Acesso a RAM/arquivo continua necessário.
- `-GpuCacheMiB 4096` é o orçamento de **pesos**, não o consumo total de VRAM.
- O KV FP32 do 30B para contexto 4096 ocupa 768 MiB de valores na GPU. Some bancos dos especialistas, ativações, driver e desktop.
- `-GpuKvMiB 1024` é um limite de segurança para KV, separado do cache de pesos. Um contexto que o exceda é recusado. Aumentá-lo explicitamente não comprova disponibilidade de VRAM.
- `mmap` permite acesso ao arquivo, mas paginação/leituras frequentes do SSD podem limitar muito a velocidade nos 16 GB de RAM do PC.
- Reduza `-Context` se o prompt couber num contexto menor; isso reduz a reserva KV, sem garantir aceleração de todos os componentes. Reduza o cache de pesos em caso de falta de VRAM; mais transferências podem tornar a geração mais lenta.

Não substitua, trunque ou edite o GGUF enquanto estiver mapeado pelo programa.

## CLI

```powershell
cargo build --release --locked
.\target\release\qwen-rust.exe devices
.\target\release\qwen-rust.exe inspect --model 'D:\Modelos\modelo.gguf' --ctx 4096
.\target\release\qwen-rust.exe chat --model 'D:\Modelos\modelo.gguf' --tokenizer models/tokenizer.json --prompt 'Explique borrowing.' --adapter 5700
.\target\release\qwen-rust.exe bench --model 'D:\Modelos\modelo.gguf' --tokens 151644,894,198,2683 --ctx 4096 --compare --runs 3
```

`--vulkan-reference` seleciona o caminho diagnóstico. `--device cpu` executa a referência CPU. Adaptadores por software exigem `--allow-software`; não há fallback silencioso. `--raw` fornece texto sem wrappers ChatML. Texto vai para stdout e progresso/métricas para stderr.

Aceita GGUF v2/v3 único, arquitetura `qwen3moe`, Q/K/V separados e tipos F32, F16, Q4_0, Q8_0, Q4_K, Q5_K e Q6_K. Sem shards, IQ/Q2/Q3, modelos dense/Next/VL, LoRA, treinamento ou RoPE escalado/YaRN. As verificações recusam configurações não implementadas.

## Testes e CI

```powershell
.\Verificar.ps1
.\Verificar.ps1 -GPU
```

A CI verifica formatação, Clippy, testes CPU e compilação release no Linux/Windows; o job Linux com Mesa executa os testes Vulkan, a referência NumPy e a comparação de 48 camadas. A passagem desses jobs não é um teste na RX 5700 XT nem do modelo real 30B.

`tests/data/tiny-qwen3moe.gguf` contém apenas pesos sintéticos de duas camadas. Testes geram temporariamente os tensores de quantização mista. Para reproduzir o cenário sintético de 48 camadas com Python/NumPy:

```powershell
python tools/reference_fixture.py --out test-output/dispatch --profile dispatch
.\target\release\qwen-rust.exe bench --model test-output/dispatch/tiny-qwen3moe.gguf --tokens 1,5,2,9 --ctx 16 --gpu-cache-mib 128 --compare --runs 5
```

Veja [OPTIMIZATION.md](OPTIMIZATION.md) para limites, contrato dos contadores e próximas otimizações.

## Código e referências

| Arquivo | Responsabilidade |
| --- | --- |
| `src/gpu_graph.rs`, `shaders/ops.wgsl` | Grafo residente, atenção, KV e operações vetoriais |
| `src/backend.rs`, `shaders/matvec.wgsl` | Cache de pesos, matvec quantizado e em lote, referência Vulkan |
| `src/model.rs`, `src/ops.rs`, `src/quant.rs` | Forward e operações CPU de referência |
| `src/sampling.rs`, `src/main.rs` | Amostragem, tokenizer, ChatML e CLI |
| `src/benchmark.rs`, `Benchmark.ps1` | Comparação de execução e paridade |
| `scripts/Common.ps1`, `Rodar.ps1`, `Verificar.ps1` | Preparação, execução e verificações Windows |

Modelo/configuração: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct . Tokenizer fixado em `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`, SHA-256 `19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d`.

Interoperabilidade GGUF/GGML: https://github.com/ggml-org/llama.cpp/tree/master/ggml e https://github.com/ggml-org/llama.cpp/tree/master/gguf-py . Backend: https://github.com/gfx-rs/wgpu (0.19.4 fixado; `Cargo.lock` incluído).

Código do projeto: MIT. Pesos originais Qwen: Apache-2.0, com avisos próprios a preservar se redistribuídos. O repositório não redistribui os pesos reais nem o tokenizer oficial.
