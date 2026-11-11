# Validação — Qwen Rust 0.2.1

Data: 10/11/2026. Ambiente local: Linux x86_64, Rust 1.99.1, Vulkan por software Mesa llvmpipe/LLVM 19.1.7. Sem medição desta versão na RX 5700 XT.

## Verificações locais

- 20 testes passaram: 12 CPU e 8 Vulkan habilitados explicitamente.
- Clippy com `-D warnings`, formatação e build release passaram.
- Referência NumPy independente: maior erro absoluto de logits 2.38418579e-07.
- Dois testes novos cobrem buffer expulso ainda referenciado por encoder pendente, reuso depois da conclusão, proprietário Rust externo e capacidade incompatível.
- Os testes anteriores cobrem cache com orçamento zero, quantização mista, atenção longa, troca de KV, CLI, reset e os sete tipos de pesos suportados.

## Comparação controlada de reciclagem

Comando release:

```text
qwen-rust bench --model test-output/dispatch/tiny-qwen3moe.gguf --tokens 1,5,2,9 --ctx 16 --gpu-cache-mib 4 --compare-weight-reuse --warmup 1 --runs 5 --allow-software
```

Mesmos pesos sintéticos de 48 camadas, mesmo grafo residente, cache LRU de 4 MiB, quatro prefixos, precisão e operações. O cache pequeno força expulsões. Ambos os modos usam o caminho de upload da versão 0.2.1; a única diferença da comparação é o reuso desligado/ligado. Não se trata de comparação binária contra a versão 0.2.0.

| Por ciclo medido de quatro tokens | Sem reciclagem | Com reciclagem |
| --- | ---: | ---: |
| Mediana de cinco ciclos | 0.555322 s | 0.557864 s |
| Forwards/s, carga sintética | 7.203 | 7.170 |
| Misses | 5572 | 5572 |
| Bytes enviados | 26083328 | 26083328 |
| Novos buffers GPU de pesos | 5572 | 1696 |
| Buffers GPU de pesos reciclados | 0 | 3876 |
| Readbacks | 196 | 196 |
| Submissions explícitas | 196 | 196 |

As novas alocações explícitas de pesos caíram **69.56%** no ciclo mostrado. Hits, misses, expulsões, bytes solicitados e bytes enviados foram iguais em cada par dos cinco ciclos. Paridade: erro máximo 0.0, mesmas escolhas greedy.

**Não houve ganho de tempo de parede demonstrado neste ambiente:** razão entre medianas sem/com reciclagem 0.9954×, próxima de 1. O resultado comprova a redução de alocações e a preservação numérica, sem prever tokens/s na Radeon. As alocações de staging e trabalho interno do driver/wgpu não são contabilizadas como buffers de pesos.

Relatório completo: [weight-reuse-benchmark-v0.2.1.json](validation/weight-reuse-benchmark-v0.2.1.json).

## Diagnóstico novo

`weight_prepare_ms` mede a rotina de cache/upload e `gpu_readback_wait_ms` mede o trecho bloqueado no readback. São tempos de parede de CPU, não timestamps dos kernels nem medidas de leitura do SSD. O programa separa `decode_stats` de `prefill_stats` e informa os bytes de upload por passo e a taxa de acerto por bytes.

A política de cache, o orçamento de pesos, a precisão, o número de especialistas e a arquitetura foram preservados. Um buffer só é reutilizado depois de confirmação explícita de conclusão GPU, com mesma capacidade e sem outro proprietário Rust. Não há pool extra que dobre o orçamento de VRAM.

## CI e validação no hardware

A CI executa Linux, Windows/PowerShell e Vulkan por software. Também confere a paridade e a redução de alocações sob pressão de cache, sem impor um limiar de velocidade dependente do host. Consulte o status do commit no PR para saber o resultado remoto.

No PC do operador, após atualizar o código:

```powershell
.\Verificar.ps1 -GPU
.\Benchmark.ps1 -Adapter '5700' -CompareWeightReuse -OutFile 'benchmark-weight-reuse.json'
```

Compare também a mesma geração natural com `Rodar.ps1` e com `Rodar.ps1 -NoWeightReuse`, mantendo contexto, prompt, seed e tamanho da resposta. Cache do SO, ordem das execuções e comportamento do driver podem alterar o tempo.

Histórico: [0.2.0](validation/BASELINE_v0.2.0.md) e [0.1.0](validation/BASELINE_v0.1.0.md). Os resultados sintéticos de versões anteriores não representam o desempenho do 30B real na RX 5700 XT.
