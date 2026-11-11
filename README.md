# Qwen Rust 0.1.0 — motor próprio experimental

Inferência de Qwen3 MoE em Rust, com multiplicação de matrizes quantizadas em Vulkan por shaders WGSL. Reutiliza os pesos treinados do Qwen3-Coder-30B-A3B-Instruct. Não usa Ollama, llama.cpp, PyTorch ou um servidor externo para inferência.

**Estado:** implementação experimental do fluxo de geração completo, validada com pesos sintéticos. A execução dos pesos reais de 30B e o desempenho na RX 5700 XT ainda precisam ser validados. Não é uma substituição de produção do Ollama, nem há ganho de velocidade comprovado.

## O que foi implementado

- Leitor GGUF v2/v3 com mmap, verificação de limites, alinhamento e dimensões.
- F32, F16, Q4_0, Q8_0, Q4_K, Q5_K e Q6_K; cobre os tipos encontrados na distribuição Ollama Q4_K_M examinada.
- Embeddings, RMSNorm, normalização de Q/K, RoPE NeoX, atenção causal GQA e cache KV.
- Roteamento MoE por top-k com renormalização, SwiGLU dos especialistas e conexões residuais.
- Projeção do vocabulário, tokenização oficial, ChatML de uma rodada, EOS, saída incremental e amostragem com seed, temperatura, top-k e top-p.
- Backend CPU de referência e backend Vulkan para matrizes, sem fallback silencioso.
- Cache LRU de pesos compactados na GPU e divisão de matrizes quando excedem o limite de buffer do adaptador.
- Comandos `devices`, `inspect`, `logits` e `chat`.

## Distribuição real do trabalho

| Operação | Backend Vulkan selecionado |
| --- | --- |
| Q/K/V, projeção de atenção, router e matrizes dos especialistas | GPU, com pesos quantizados |
| Projeção final do vocabulário | GPU |
| Leitura de embeddings, normalizações, RoPE e residuais | CPU |
| Atenção sobre o histórico e cache KV FP32 | CPU / RAM |
| Seleção dos especialistas, tokenização e amostragem | CPU |

O código de inferência é Rust e os kernels são WGSL; `wgpu` acessa Vulkan e o driver da placa. Isso não significa que todas as operações já ficam na GPU.

## Executar no Windows

Pré-requisitos: Rust stable atualizado, Visual Studio Build Tools com ferramentas MSVC para C++ e Windows SDK, e driver AMD com Vulkan. A compilação/testes locais deste pacote usaram Rust 1.99.1 no Linux; Windows e PowerShell não foram executados neste ambiente. O Build Tools é usado pelo toolchain; o motor de inferência não incorpora llama.cpp.

Extraia o ZIP e abra o PowerShell na pasta `qwen-rust`:

```powershell
rustup update stable
rustup component add rustfmt clippy
.\Verificar.ps1 -GPU
.\Rodar.ps1 -Prompt 'Crie uma funcao Rust para ordenar uma lista.'
```

`Rodar.ps1` procura o manifesto local de `qwen3-coder:30b` e abre seu blob GGUF existente. Não chama o Ollama nem altera o modelo. Se `OLLAMA_MODELS` estiver definido, usa esse diretório. Caso não encontre o arquivo:

```powershell
.\Rodar.ps1 -Model 'D:\Modelos\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf' -Prompt 'Explique ownership em Rust.'
```

O script compila em release, valida o GGUF, lista adaptadores e baixa somente o `tokenizer.json` oficial, fixado por revisão e SHA-256. Os pesos reais de 30B não estão no pacote. O programa verifica a correspondência de cada token do tokenizer com a tabela GGUF; posições extras de padding no vocabulário do modelo são mascaradas na amostragem.

Para selecionar a placa, se houver mais de um adaptador:

```powershell
.\Rodar.ps1 -Adapter '5700' -GpuCacheMiB 4096 -MaxTokens 64
```

O cache padrão de 4096 MiB é apenas o orçamento de pesos na GPU. Buffers de trabalho, o driver e o desktop consomem memória adicional. Reduza o cache se houver falta de VRAM; reduzir pode aumentar transferências e lentidão.

## Uso direto da CLI

```powershell
cargo build --release --locked
.\target\release\qwen-rust.exe devices
.\target\release\qwen-rust.exe inspect --model 'D:\Modelos\modelo.gguf' --ctx 4096
.\target\release\qwen-rust.exe chat --model 'D:\Modelos\modelo.gguf' --tokenizer '.\models\tokenizer.json' --prompt 'O que e borrowing?' --device vulkan --ctx 4096 --max-tokens 128
.\target\release\qwen-rust.exe logits --model 'D:\Modelos\modelo.gguf' --tokens 151644,894 --device vulkan
```

`--device cpu` executa a referência lenta na CPU. `--allow-software` é uma opção explícita de diagnóstico para Vulkan por software, não aceleração física. `--raw` desativa os wrappers ChatML. A saída de texto vai para stdout; progresso, identificação do adaptador e métricas vão para stderr.

O limite `--ctx` cobre prompt e geração. A CLI recusa o pedido se `tokens_do_prompt + max_tokens > ctx`. O modelo permanece mapeado em modo de leitura: não substitua, trunque ou edite o arquivo enquanto o processo estiver aberto.

## Limites materiais desta versão

- O arquivo examinado tem 18.556.688.736 bytes. Ele não cabe integralmente nos 8 GB de VRAM da RX 5700 XT. São necessários RAM e acesso ao arquivo; os 16 GB de RAM do PC também serão disputados pelo Windows e pelo restante do programa.
- `mmap` não transforma o SSD em memória rápida. Se páginas forem continuamente descartadas e relidas, a inferência pode ficar muito lenta.
- O cache KV usa FP32 na RAM: 768 MiB de valores para contexto 4096 no 30B, além de capacidades das alocações e demais dados.
- Prefill sequencial, sincronização e retorno à CPU depois de cada matvec. No modelo de 48 camadas/top-8 isso representa 1393 matvecs por token. Pode ser substancialmente mais lento que motores maduros.
- Sem atenção GPU, prefill em lote, FlashAttention, cache KV quantizado, kernels MoE agrupados ou retenção das ativações na GPU nesta versão.
- Aceita GGUF único de arquitetura `qwen3moe`, com tensores separados de Q/K/V e nomes padronizados. Não implementa shards, arquiteturas densas/Next/VL, outros formatos IQ/Q2/Q3, LoRA, treinamento, API HTTP ou RoPE escalado/YaRN.
- O cabeçalho e o tokenizer do modelo real foram conferidos. Não houve geração com os pesos treinados reais de 30B, nem medição na RX 5700 XT. O benchmark antigo do Ollama não é resultado deste motor.

## Verificação

```powershell
cargo test --locked
cargo test --locked -- --ignored
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check
```

Os testes marcados `ignored` precisam de Vulkan e são executados explicitamente pelo segundo comando. Por padrão recusam adaptadores classificados como CPU. Para validar intencionalmente com um Vulkan por software, defina `QWEN_TEST_ALLOW_SOFTWARE=1` somente no ambiente de teste.

`tests/data/tiny-qwen3moe.gguf` contém pesos aleatórios sintéticos de duas camadas, criados para teste; não é um modelo treinado. Os logits esperados vêm de uma implementação NumPy independente em `tools/reference_fixture.py`. Os testes Rust comuns não dependem de Python. Para regenerar/conferir a referência, use Python com NumPy:

```powershell
python tools/reference_fixture.py --binary target/release/qwen-rust.exe --device cpu
python tools/reference_fixture.py --binary target/release/qwen-rust.exe --device vulkan
```

Consulte `VALIDACAO.md` para resultados e alcance exato da validação.

## Estrutura

| Arquivo | Responsabilidade |
| --- | --- |
| `src/gguf.rs` | Metadados, tensores, validação e mapeamento do GGUF |
| `src/quant.rs` | Descompactação CPU de referência |
| `src/backend.rs` | Matvec CPU/Vulkan, cache, transferências e métricas |
| `shaders/matvec.wgsl` | Descompactação e redução das matrizes na GPU |
| `src/ops.rs` | RMSNorm, RoPE, atenção e seleção MoE |
| `src/model.rs` | Configuração e forward causal completo |
| `src/sampling.rs` | Seleção do próximo token |
| `src/main.rs` | CLI, tokenizer, ChatML e geração |
| `tests/` | Layouts quantizados, referência numérica, parser, cache e CLI |
| `Rodar.ps1` / `Verificar.ps1` | Execução e verificação no Windows |

## Fontes e pesos

- Modelo/configuração: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- Revisão do tokenizer: `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`.
- SHA-256 do tokenizer: `19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d`.
- Layouts GGML/GGUF e nomes dos tensores: https://github.com/ggml-org/llama.cpp/tree/master/ggml e https://github.com/ggml-org/llama.cpp/tree/master/gguf-py . Referências de interoperabilidade; não são dependências de execução.
- Backend Rust/Vulkan: https://github.com/gfx-rs/wgpu . Dependência fixada em `0.19.4` e `Cargo.lock` incluído para reproduzir o build validado.

O código deste projeto usa MIT. Os pesos originais Qwen têm licença própria Apache-2.0; preserve seus avisos ao redistribuí-los. O pacote não redistribui os pesos reais nem o tokenizer oficial.
