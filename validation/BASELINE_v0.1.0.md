# Validação — Qwen Rust 0.1.0

Data: 10/11/2026. Ambiente: Linux x86_64, Rust 1.99.1, Cargo.lock incluído. Vulkan de teste: Mesa llvmpipe, LLVM 19.1.7, dispositivo classificado como CPU. Não havia uma RX 5700 XT acessível.

## Resultados

| Verificação | Resultado | Alcance |
| --- | --- | --- |
| `cargo build --release --locked` | Passou | Compilação Linux otimizada |
| `cargo fmt --check` | Passou | Código Rust |
| `cargo clippy --locked --all-targets -- -D warnings` | Passou | Código e testes Rust |
| `cargo test --locked` | 9 passaram; 3 Vulkan ignorados por padrão | CPU, quantização, modelo, GGUF e CLI |
| `cargo test --locked -- --ignored` com Vulkan software explícito | 3 passaram | Shaders, cache, grid e forward sintético |
| Referência independente NumPy / CPU release | Passou | Quatro prefixos de 1–4 tokens, modelo sintético de duas camadas |
| Referência independente NumPy / Vulkan release | Passou | Mesmo modelo sintético, kernels executados em llvmpipe |
| Cabeçalho real `qwen3-coder:30b` | Passou | 579 tensores, dimensões, tipos suportados e limites do arquivo |
| Tokenizer oficial vs tabela do GGUF real | Passou | Correspondência de tokens; teste interrompido antes do forward por limite deliberado de contexto |
| Compilação/executável Windows e scripts PowerShell | Pendente | Não houve Windows/PowerShell neste ambiente |
| Geração com pesos treinados reais de 30B | Pendente | Não executada |
| Benchmark de desempenho na RX 5700 XT | Pendente | Não executado |

## Erro numérico observado

Modelo de teste: duas camadas, hidden 8, 4 cabeças Q, 2 cabeças KV, dimensão de cabeça 4, 4 especialistas, top-2 e vocabulário 16. Pesos FP32 aleatórios com seed fixa. O tamanho Q é diferente do hidden, como também ocorre no Qwen3-Coder-30B.

| Prefixo de tokens | Maior erro absoluto CPU | Maior erro absoluto Vulkan software |
| --- | --- | --- |
| `[1]` | 1,1921 × 10⁻⁷ | 1,7881 × 10⁻⁷ |
| `[1,5]` | 1,4901 × 10⁻⁷ | 1,6391 × 10⁻⁷ |
| `[1,5,2]` | 1,7881 × 10⁻⁷ | 2,0862 × 10⁻⁷ |
| `[1,5,2,9]` | 2,3842 × 10⁻⁷ | 1,1921 × 10⁻⁷ |

Asserção utilizada: `rtol=2e-4`, `atol=2e-5`. Não se pode extrapolar esse erro para 48 camadas, contexto longo e pesos reais quantizados sem uma comparação específica.

## Testes cobertos

1. Seleção top-k dos especialistas, pesos renormalizados e recusa de logits inválidos.
2. Rotação NeoX com pares entre metades do vetor.
3. Mapeamento GQA de cabeças Q para KV.
4. Amostragem greedy, seed reproduzível e parâmetros inválidos.
5. Recusa de tipos e dimensões quantizadas incompatíveis.
6. Layouts F32, F16, Q4_0, Q8_0, Q4_K, Q5_K e Q6_K com escalas/valores não uniformes e blocos consecutivos.
7. Forward CPU completo comparado com NumPy, cache KV, reset e limite de contexto.
8. Cabeçalhos malformados, contagens excessivas e arquivos truncados.
9. CLI de chat com tokenizer compatível e recusa de marcadores especiais no conteúdo.
10. Matvec Vulkan para os sete formatos e reutilização do cache.
11. Grid Vulkan com 70.001 linhas e descarte LRU por orçamento.
12. Forward Vulkan completo contra a referência e invalidação do cache ao trocar de modelo.

Os testes de formatos usam equações de referência separadas dos decodificadores de produção, mas não constituem validação de todos os possíveis padrões binários. Não houve comparação dos logits reais com llama.cpp/Transformers nesta versão.

## Inspeção do GGUF real

Manifesto consultado em `https://registry.ollama.ai/v2/library/qwen3-coder/manifests/30b`:

- Digest do blob: `sha256:1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a`.
- Tamanho total declarado: 18.556.688.736 bytes; pesos compactados: 18.550.716.416 bytes.
- Arquitetura: `qwen3moe`; camadas: 48; hidden: 2048.
- Cabeças Q/KV: 32/4; dimensão: 128.
- Especialistas: 128; escolhidos por token: 8; tamanho interno por especialista: 768.
- Vocabulário dos embeddings: 151.936; tokenizer oficial: 151.669 tokens.
- RoPE theta: 10.000.000; contexto nativo informado: 262.144.
- KV FP32 calculado para contexto 4096: 805.306.368 bytes = 768 MiB de valores.

A inspeção baixou somente os primeiros 16 MiB via HTTP Range. Um arquivo esparso temporário com o tamanho declarado permitiu verificar offsets e dimensões; o restante era zero e **não foi usado para inferência**. Esse arquivo não faz parte do pacote. A inspeção não valida a integridade de todos os pesos nem o SHA-256 do blob completo.

Os metadados atuais do registro podem diferir de um arquivo baixado anteriormente. `inspect` verifica o arquivo efetivamente informado pelo usuário.

## Critérios ainda necessários para concluir a validação do 30B

1. Executar `Verificar.ps1 -GPU` no PC e confirmar o adaptador físico escolhido.
2. Executar o arquivo real com tokenizer correspondente e comparar logits de prompts curtos com uma referência do mesmo GGUF.
3. Medir prefill, tempo até primeiro token, geração, VRAM, RAM e bytes transferidos usando o mesmo contexto e os mesmos pesos da comparação com Ollama.
4. Só atribuir ganhos às otimizações que tenham passado por essa comparação. A sincronização por matvec e a atenção na CPU são limitações conhecidas da versão 0.1.0.

Os resultados JSON brutos das comparações release e da inspeção estão em `validation/`.
