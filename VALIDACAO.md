# Validação — Qwen Rust 0.2.0

Data: 10/11/2026. Ambiente local: Linux x86_64, Rust 1.99.1, Mesa llvmpipe/LLVM 19.1.7, Vulkan **por software**. Não houve acesso à RX 5700 XT do operador.

## Resultado local

- Compilação release passou, com Cargo.lock.
- Formatação e Clippy com `-D warnings` passaram.
- **18 testes passaram**, incluindo 12 CPU e 6 Vulkan habilitados explicitamente.
- Comparação NumPy independente dos quatro prefixos sintéticos de duas camadas: maior erro absoluto GPU 2.38418579e-07.
- O benchmark residente de 48 camadas passou na comparação de todos os logits com o caminho de referência. Maior erro absoluto: 9.53674316e-07; escolhas greedy iguais.
- A CI inclui Linux, Windows, parsing de PowerShell e testes Vulkan por software. Consulte o status do commit/PR para o resultado remoto; a aprovação da CI não é benchmark do hardware do operador.

## Benchmark sintético de 48 camadas

Comando executado em release:

```text
qwen-rust bench --model test-output/dispatch/tiny-qwen3moe.gguf --tokens 1,5,2,9 --ctx 16 --gpu-cache-mib 128 --compare --warmup 1 --runs 5 --allow-software
```

Perfil: hidden 32, 48 camadas, 128 especialistas/top-8, FF 32, Q 8 cabeças × dimensão 8, KV 2 cabeças, vocabulário 256. Pesos FP32 aleatórios, não treinados. O arquivo tem cerca de 74 MiB; o conjunto de pesos acessado nesses prefixos cabe no cache de 128 MiB. Ambos os modos medidos tiveram **zero uploads de pesos** depois do aquecimento. Isso é favorável ao cache e não representa necessariamente o 30B maior que a VRAM.

| Medida por ciclo de quatro tokens | Referência | Residente |
| --- | ---: | ---: |
| Mediana de cinco ciclos | 1.291845 s | 0.507259 s |
| Forwards/s nessa carga sintética | 3.096 | 7.886 |
| Matrizes lógicas | 5572 | 5572 |
| Dispatches GPU totais | 5572 | 3656 |
| Submissions explícitas | 5572 | 196 |
| Readbacks síncronos | 5572 | 196 |
| Bytes retornados à CPU | 790528 | 102400 |
| Alocações explícitas de buffers de trabalho | 22288 | 0 |

Razão entre medianas: **2.547×** nesse ambiente e carga. Readbacks por token: 1393 → 49, redução de 96.48%. Os números são evidência local da mudança de execução, não promessa de aceleração na RX 5700 XT. Variações entre execuções são esperadas no driver por software.

O caminho de referência é o modo diagnóstico do mesmo executável 0.2, com atenção/vetores CPU e readback por matriz, como na execução 0.1. Ele também recebe a melhoria geral de cache LRU; não é uma cópia binária do executável 0.1.

Dados completos: [dispatch-benchmark-v0.2.0.json](validation/dispatch-benchmark-v0.2.0.json). As métricas de prefill do chat omitem as projeções intermediárias descartadas; este benchmark solicita logits em cada prefixo para verificar paridade e não mede essa economia adicional.

## Testes de regressão

1. Seleção e renormalização MoE, valores inválidos.
2. RoPE NeoX entre metades.
3. Mapeamento GQA.
4. Seed, greedy e parâmetros da amostragem.
5. Top-k parcial contra a distribuição de ordenação completa, incluindo empates.
6. Dimensões e tipos quantizados inválidos.
7. Layouts F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K.
8. Matvec Vulkan desses sete tipos.
9. Grid com 70.001 linhas e expulsão LRU.
10. Forward CPU contra NumPy, reset e limites.
11. Cabeçalhos GGUF malformados e truncados.
12. CLI de chat, tokenizer e marcadores de controle.
13. Forward residente contra NumPy e troca de modelo.
14. Prefill sem projeção preservando estado causal.
15. Buffers persistentes, submissions/readbacks, cache com orçamento zero, limite de KV e recusa de troca de local de KV sem reset.
16. Tensores quantizados misturados comparados entre CPU, Vulkan referência e residente, com expulsão de pesos.
17. Atenção com 80 posições, incluindo a transição de 64 posições.
18. CLI de benchmark e contagem de forwards medidos.

## Ainda precisa ser medido no PC

- Executar `Verificar.ps1 -GPU` e `Benchmark.ps1` no Windows com o arquivo real de 30B.
- Conferir o adaptador físico e a paridade nos logits. O relatório identifica driver/software, orçamento de pesos e KV.
- Medir também geração natural, tempo até primeiro token, VRAM, RAM, paginação e uploads de pesos. Prefixos fixos quentes podem favorecer o cache.
- Não foi alterada a quantização dos pesos nem reduzido o número de camadas ou especialistas. FP32 é preservado para ativações/KV, com diferenças de ordem das somas paralelas.

O histórico da inspeção do cabeçalho real e validação 0.1 permanece em [BASELINE_v0.1.0.md](validation/BASELINE_v0.1.0.md). A geração real da 0.1 foi informada pelo operador; não equivale à validação da nova versão na placa.
