# Otimização e contrato de medição — 0.2

## Implementado sem aproximações adicionais

1. **Estado residente:** cada token entra com a linha de embedding; ativações e KV ficam na GPU. Por camada, a CPU recebe apenas os 128 logits do router, escolhe os oito especialistas e providencia seus pesos. Ao final, recebe os logits do vocabulário. O lote de especialistas e a atenção seguinte compartilham o mesmo encoder até a próxima decisão de roteamento.
2. **Especialistas em lote:** cópias entre buffers GPU juntam os pesos selecionados em três bancos reutilizáveis. Gate/up/down usam a dimensão Z do dispatch para oito especialistas. Os resultados são combinados na GPU com as probabilidades renormalizadas.
3. **Precisão:** normalizações, atenção, KV e ativações permanecem FP32. Os pesos continuam na quantização do arquivo original. A ordem paralela das somas pode alterar números; asserções usam tolerâncias, não igualdade binária.
4. **Alocações:** buffers de trabalho e uniformes de formato persistem. Os parâmetros de posição são escritos uma vez por token. Misses do cache de pesos ainda podem alocar buffers; não há promessa de zero alocações totais.
5. **LRU:** heap de idades com verificação de entradas antigas e compactação. A contabilidade cobre os buffers no cache. Comandos já submetidos podem manter recursos expulsos vivos até terminarem; o orçamento não é um limite rígido de consumo total da GPU.
6. **Prefill:** as projeções de vocabulário e readbacks finais são omitidos nos tokens intermediários. O último token produz os logits necessários à geração. As camadas de atenção ainda são processadas sequencialmente token a token.
7. **Amostragem:** greedy usa busca linear; top-k usa seleção parcial. O subconjunto final mantém a ordenação e a mesma sequência pseudoaleatória da referência.

O `Model` exige reset antes de alternar entre KV na CPU e KV residente; recusa a troca em uma sequência ativa para evitar histórico incompleto.

## Contadores

| Campo | Significado |
| --- | --- |
| `matvecs` | Matrizes lógicas avaliadas. Um dispatch de oito especialistas conta oito. Tiling pode aumentar a contagem. |
| `gpu_dispatches` | Dispatches de compute efetivamente registrados; inclui kernels vetoriais/atenção no modo residente. |
| `submissions` | Chamadas explícitas a `queue.submit`; uploads podem gerar trabalho adicional dentro do wgpu. |
| `readbacks` / `readback_bytes` | Mapeamentos síncronos de staging para a CPU e bytes solicitados. |
| `uploaded_bytes` | Bytes de pesos compactados enviados por misses do cache, com padding. Não inclui embedding, uniformes ou rates. |
| `cache_hits/misses/evictions` | Eventos do cache de pesos. Não são leituras do SSD nem page faults do SO. |
| `workspace_allocations` | Buffers de trabalho/normas/uniformes criados explicitamente pelo código. Exclui pesos e alocações internas do driver/wgpu. |
| `matvec_ms` | Tempo de parede de matvec no caminho de referência; inclui espera e transferências. Zero no grafo residente. |
| `device_forward_ms` | Tempo de parede do token no grafo residente; inclui roteamento CPU e transferências. Não é tempo puro dos kernels. |

No 30B de 48 camadas/top-8, com tensores cabendo nos limites por binding e solicitando logits: 1393 matrizes lógicas, 385 dispatches de matvec e 529 operações vetoriais/atenção = 914 dispatches no modo residente. Há 49 submissions explícitas e 49 readbacks. O modo referência usa 1393 submissions/readbacks. O prefill intermediário residente mantém 49 submissions, executa 48 readbacks e omite a normalização/projeção final do vocabulário. Tiling e configurações diferentes alteram esses números.

## O que continua limitando o desempenho

- Todos os pesos não cabem na RX 5700 XT. O padrão de especialistas muda com o texto; benchmarks quentes com prefixos fixos podem ter muito mais cache hits que uma conversa longa.
- Ainda há 48 decisões síncronas de roteamento por token. Carregar todos os especialistas na GPU evitaria essas decisões na CPU, mas exige mais VRAM que a placa oferece para esse arquivo.
- Os bancos de especialistas exigem cópias VRAM→VRAM. Esse custo pode compensar a redução de dispatches de forma diferente em cada GPU.
- Atenção GPU desta versão é uma implementação FP32 de referência, sem FlashAttention, fusão avançada ou ajuste específico para RDNA.
- O matvec genérico descompacta elementos e faz redução por workgroup. Kernels especializados por quantização, layouts e wavefront podem melhorar o resultado; precisam de benchmark na placa.
- O benchmark local é software Vulkan, portanto mede uma combinação de CPU/driver/overhead. Não prevê tokens/s do 30B quantizado na Radeon.

## Próximas etapas, escolhidas por medição

| Prioridade | Mudança candidata | Evidência necessária |
| --- | --- | --- |
| 1 | Perfil por etapa, tipos de matrizes e misses de especialistas no 30B real | Relatório do `Benchmark.ps1` e geração natural |
| 2 | Kernels Q4_K/Q6_K específicos e fusão QKV | Paridade por tensor e redução do tempo de matrizes na RX 5700 XT |
| 3 | Prefill em lote e atenção fundida | Testes causais, diferentes comprimentos e menor tempo até primeiro token |
| 4 | Cache de pesos com partição dense/experts e agendamento | Menos uploads/paging sem exceder VRAM |
| 5 | KV FP16/quantizado, menor quantização dos pesos ou decodificação especulativa | Opções explícitas, avaliação de qualidade, memória e velocidade; nada habilitado silenciosamente |

Trocar Rust por outra linguagem só é justificável se uma implementação medida superar o caminho atual ou reduzir o custo de manutenção. Trocar a arquitetura transformer altera a compatibilidade com os pesos e pode exigir adaptação/treinamento. Não é uma otimização transparente do mesmo modelo.

A versão 0.2 não afirma atingir o limite físico da placa nem a velocidade máxima possível do modelo. O alvo é uma melhora reproduzível, mantendo um caminho de referência e resultados auditáveis.
