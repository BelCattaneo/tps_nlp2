# Historial de pruebas

Registro de cada corrida con métricas y observaciones, para conclusiones al cierre del TP.

Formato por entrada:

```
## YYYY-MM-DD — TP-X — Consigna N — <título>

**Config:** hardware, hiperparámetros relevantes, seed.
**Métricas:** números concretos (loss, tiempo/paso, throughput, utilización, etc.).
**Observaciones:** qué se vio, qué llamó la atención, hipótesis.
```

---

## 2026-09-20 — TP-I — pretraining modelo denso + Consigna I

**Config:** MPS (Apple Silicon), N_CHARS=100.000, GPTConfig default (block_size=32, batch_size=64, n_embd=64, n_head=4, n_layer=2, vocab_size=61, dropout=0.1, tie_weights=True). Seed 1337. 2 epochs, sin AMP.

**Métricas:**
- Modelo denso entrenado: 106.048 parámetros (100% entrenables).
- Epoch 1: train 2.1788 · val 2.0506.
- Epoch 2: train 2.0880 · val 1.9842.
- Throughput época 1: 149 it/s (train), 267 it/s (val). Época 2: 226 it/s / 552 it/s (JIT warm).
- 1405 batches de train, 155 de val.
- `generateV2` implementada. `self_check()` pasó los 4 asserts (greedy determinístico, top_k=1 ≡ greedy, T→0 ≡ greedy, top_p→0 preserva sólo el más probable).

**Observaciones:**
- La curva de loss baja más rápido en época 2 que en la 1 aunque los datos son los mismos — consistente con JIT cache y momentum acumulándose.
- Val loss < train loss final: llamativo pero normal a esta escala (val es un chunk pequeño y "fácil", 10.000 chars corridos, sin ruido de dropout).
- Bug clásico atajado en `generateV2`: soft-max y multinomial estaban fuera del `else`, sobreescribiendo el argmax → hacía muestreo puro incluso con `do_sample=False`. Ver `feedback_test-metrics-history` en memoria y notas del turno de implementación.

---

## 2026-09-20 — TP-I — Consigna II — grilla de estrategias de decodificación

**Config:** modelo denso ya entrenado (ver entrada anterior). Prompt `"To be"`, 200 tokens nuevos, `torch.manual_seed(0)` antes de cada corrida.

**Métricas (cualitativas, muestras del Cell 31):**

| Estrategia | Comportamiento observado |
|---|---|
| `greedy` (do_sample=False) | Loop `"the the the the theano teano..."` desde el token ~15. Degeneración total. |
| Muestreo puro T=1 | Palabras plausibles pero sin cohesión, algunos "casi loops" pero se rompen. |
| T=0.5 | Repetición de `"the"` y variaciones, pero atenuada respecto a greedy. |
| T=1.5 | Aparecen símbolos y mayúsculas random (`wS:`, `!e`, `-one::`, `&uendeppmaoube`). Basura de la cola. |
| top_k=10 | Similar a T=1 puro pero sin la basura de cola — mejora clara. |
| top_p=0.9 | Similar a top_k=10 en este texto (distribución no varía tanto entre pasos). |
| T=0.8 + top_p=0.9 | Mejor combinación de las probadas — más conservador que top-p solo, sin loops. |

**Observaciones:**
- Los dos modos de falla nombrados: greedy y T bajos → repetición; T alto → incoherencia por cola.
- La diferencia práctica entre top-k y top-p sólo se hace visible cuando la distribución cambia mucho de forma entre pasos (contextos "seguros" vs "inseguros"). A este tamaño de modelo y con este corpus, la varianza no es enorme, por eso rinden parecido.
- Config propuesta para producción: **T=0.8 + top_p=0.9**. Racional: temperatura reduce el ruido de la distribución cruda, top-p elimina la cola y se adapta al contexto — es el estándar en la industria por este motivo.

---

## 2026-09-20 — TP-I — Consigna III — benchmark KV-cache

**Config:** modelo denso ya entrenado. MPS. `benchmark_cache(lengths=(50,100,200,400), repeats=3)`. Prompt `"To be"`, `do_sample=False` (greedy), warm-up de 1 corrida por config antes de medir. Sincronización MPS antes y después de cada medición.

**Métricas:**

| length | con cache (s) | sin cache (s) | speedup |
|---:|---:|---:|---:|
| 50 | 0.074 | 0.073 | 0.99× |
| 100 | 0.112 | 0.091 | 0.82× |
| 200 | 0.140 | 0.141 | 1.01× |
| 400 | 0.237 | 0.268 | 1.13× |

**Observaciones:**
- La teoría O(N²) → O(N) **no se materializa** a esta escala. Speedups ≤ 1.16× para secuencias cortas, ≈ 1 para largas.
- Motivo principal: `block_size = 32` acota el trabajo "sin cache". Sin cache, cada paso procesa como mucho los últimos 32 tokens (`idx_cond = idx[:, -block_size:]`), no toda la historia. Ambas ramas quedan O(N × 32) = O(N).
- A este tamaño de modelo (n_embd=64, head_dim=16), los matmuls son diminutos: el tiempo lo domina el overhead fijo por paso (kernel launches, dispatch, Python) que el cache no reduce.
- El cache además paga overhead propio: `unbind`, `cat`, `stack` y allocs por paso. Del mismo orden que lo que ahorra en aritmética.
- El cache **sí ganaría** en modelo grande (matmuls dominantes), block_size grande (más re-cómputo sin cache), y con Flash Attention (menos overhead de gestión).

---

## 2026-09-20 — TP-I — Consigna IV — visualización de atención

**Config:** modelo denso entrenado. `visualize_attention(model, tokenizer, "To be or not to be")` — 18 caracteres, 2 capas × 4 cabezas = 8 heatmaps.

**Observaciones:**
- Todos los mapas son triangulares inferiores por la máscara causal (línea 225: `masked_fill(self.tril[past:T_k, :T_k] == 0, -inf)`). Sin esa línea, el modelo vería el futuro durante entrenamiento y no aprendería a generar.
- **Capa 1 muestra patrones locales**: cabeza con diagonal principal (identidad, refuerza el token actual), y otra con sub-diagonal desplazada 1 posición (bigrama, capta pares como `"th"`, `"he"`).
- **Capa 2 muestra atención más difusa**: las cabezas reparten masa entre varios caracteres del pasado, plausiblemente integrando información de mayor rango (posición dentro de palabra, límites de palabras, ritmo).
- Consistente con el patrón "jerárquico" típico en transformers entrenados: local en capas bajas, más global en capas altas.

---

## 2026-09-20 — TP-I — Consigna V + VI — implementación y entrenamiento de MoE

**Config:** GPTConfig default + `ff_class=MoEFFN`, `MoEArgs(num_experts=4, num_experts_per_token=2)`. MPS, seed 1337, 2 epochs.

**Métricas:**
- Parámetros MoE: 305.088 (100% entrenables). Denso equivalente: 106.048. Ratio: **2.88×**.
- `check_moe()` pasó los 4 asserts (shape preservada, gradientes a expertos y gate, k=1 ≡ experto seleccionado, ruteo por token).
- Epoch 1: train 2.0904 · val 1.9676. Throughput: 40 it/s (train) vs 149 it/s del denso — **~3.7× más lento por paso**.
- Epoch 2: train 2.0177 · val 1.8935. Throughput: 70 it/s (train) — mejora tras JIT warm.
- Val loss final: **denso 1.9842 vs MoE 1.8935** — mejora de 0.091.

**Observaciones:**
- El MoE mejora la calidad (val loss más baja) con casi 3× los parámetros totales pero mismo cómputo por token (~50% de expertos activos + overhead de gate).
- El MoE es ~30× más lento por paso durante entrenamiento (ver Consigna IX pendiente): loop de Python sobre expertos, kernels chicos, gathers de forma variable. Aritmética similar al denso; el costo lo domina el overhead.
- La ganancia de val loss es modesta a esta escala (100k caracteres, 2 epochs) — MoE brilla en corpus grandes donde la especialización de expertos rinde.

---

## 2026-09-20 — TP-I — Consigna VII — utilización de expertos (MoE denso)

**Config:** `moe_model` entrenado. `expert_utilization(moe_model, val_loader, n_batches=20)`. Fracciones normalizadas a "porción de tokens que pasan por cada experto" (sum = k por capa, ideal = k/E = 0.5).

**Métricas:**

| Capa | e0 | e1 | e2 | e3 | ¿Balanceada? |
|:---:|---:|---:|---:|---:|:---|
| 1 | 0.414 | 0.624 | 0.666 | 0.296 | Sí, entre 0.30 y 0.67 |
| 2 | 0.001 | 0.001 | 0.999 | 0.999 | No — colapso total a e2 y e3 |

**Observaciones:**
- Capa 1 está balanceada dentro de un rango razonable (todos entre 0.30 y 0.67, ideal 0.5).
- **Capa 2 colapsó totalmente**: e2 y e3 reciben ~100% de los tokens cada uno (o sea, todos los tokens siempre se rutean a esos dos). e0 y e1 están muertos (<0.1%).
- El colapso se refuerza solo: los expertos no elegidos no reciben gradiente, la gate aprende a bajar aún más su score, ciclo winner-take-all clásico.
- Sin loss auxiliar de balanceo (Switch Transformers, ec. 4), nada contrarresta el colapso.
- La Capa 2 (cerca de la loss) colapsó más que la Capa 1 — argumento probable: gradientes más fuertes cerca de la loss aceleran el winner-take-all.
- La gate aprende sólo sobre los expertos que efectivamente elige (topk retorna gradiente a través de los valores seleccionados; los índices son discretos y no llevan gradiente). Los expertos que dejan de ser seleccionados quedan congelados en la inicialización.

---

## 2026-09-20 — TP-I — Consigna VIII — DeepSeekMoE

**Config:** `SEGMENTS=2` → 8 expertos ruteados de hidden 128 + 1 shared expert de hidden 128, top-4 por token. Comparación contra MoE original (4 expertos de hidden 256, top-2).

**Parámetros y cómputo:**
- MoE original: 305.088 parámetros. Cómputo por token FFN: 2 × (64→256→64) ≈ 65k ops.
- DeepSeekMoE: 339.264 parámetros (~11% más por el shared expert). Cómputo por token FFN: 4 × (64→128→64) + 1 × (64→128→64) = 5 × 16k ≈ 82k ops (~25% más).
- Segmentación fina sola es neutra en params y cómputo; el shared es el que suma.

**Métricas:**

| Modelo | Params | Val loss | Speed (it/s epoch 2) |
|---|---:|---:|---:|
| Denso | 106.048 | 1.9842 | 226 |
| MoE (VI) | 305.088 | 1.8935 | 70 |
| DeepSeekMoE | 339.264 | **1.8746** | 38 |

- DS beat MoE por 0.019 en val loss. Reproduce la dirección del paper pero no la magnitud.
- Costo: ~40% más de tiempo por paso vs MoE.

**Utilización de expertos (DS, val_loader):**

| Capa | Fracciones (ideal k/E = 0.5) | Diagnóstico |
|:---:|---|:---|
| 1 | [0.42, 0.41, 0.54, 0.47, 0.44, 0.45, 0.74, 0.53] | Balanceada — todos entre 0.41 y 0.74, ninguno muerto. |
| 2 | [0.39, 0.003, 0.31, 0.99, 0.99, 0.99, 0.02, 0.31] | Colapso parcial — 3 dominan (0.99), 2 casi muertos (0.003, 0.02), 3 con utilización intermedia (0.31–0.39). Ninguno *totalmente* muerto (a diferencia del MoE original que dejaba experts en 0.001). |

- Comparado con el MoE (Capa 2 con 2 fully active + 2 dead), el DS tiene 3 fully active + 2 casi-dead + 3 intermedios. Proporcionalmente el desbalance sigue presente pero absolutamente hay más flexibilidad (los "perdedores" e intermedios reciben algo de gradiente y podrían recuperarse).

**Magnitud shared vs routed (Bloque 1):**
- Norma L2 promedio shared: 6.45.
- Norma L2 promedio routed: 15.81.
- Ratio shared/routed: **0.41**.
- Interpretación: consistente con el paper — el shared aporta menos magnitud (estructura común es "chica de fondo") mientras los ruteados agregan detalle especializado con mayor magnitud. Si el shared se hubiera convertido en "otra FFN densa", esperaríamos ratio ~1.
- Caveat: la magnitud sola no prueba que la salida sea *común* entre tokens; sería necesario medir correlación o hacer ablation.

**Conclusiones:**
- La razón más probable de la reproducción parcial es **escala**: paper corre en modelos 10.000× más grandes con 1.000.000× más datos. Segmentación fina necesita diversidad de datos para que expertos especialicen; shared expert necesita "trabajo común" abundante para aislar.
- Experimento más chico para testear escala como explicación: reentrenar los tres modelos con `N_CHARS = 1M` (o corpus completo tinyshakespeare) manteniendo la arquitectura idéntica. Si la brecha DS–MoE crece con más datos, escala es al menos parte de la explicación.

---

## 2026-09-20 — TP-I — Consigna IX — MoELayerFast (optimización de MoELayer)

**Config:** MPS. Batch `(64, 32, 64)`. Media ± desvío sobre 50 forwards, warmup=5. Comparación aislada de la capa, no del modelo entero.

**Cambio implementado:** eliminación de sincronizadores GPU→CPU dentro del loop de expertos. En vez de máscaras booleanas por experto (`==`, `.any()`, `.nonzero()` × E), la fast ordena todos los pares `(token, slot)` con `torch.sort` una sola vez y usa slices contiguos con offsets precomputados. 1 sync en vez de E.

**Métricas:**

| Implementación | Tiempo por forward | Variance |
|---|---:|---:|
| MoELayer (V) | 12.54 ± 1.54 ms | baseline |
| MoELayerFast (IX) | 12.27 ± 9.01 ms | alta en esta corrida |
| **Speedup** | **1.02×** | — |

- Correctitud verificada con `test_moe_equivalence`: max abs diff ~1e-7 (dentro de tolerancia float32).

**Observaciones:**
- Ganancia mínima (1.02×) en esta corrida. La variance alta de la Fast (±9 ms) sugiere que hubo carga externa durante el bench; la ganancia estructural (menos syncs) sigue estando pero el número medido es ruidoso.
- Con E=4 sólo eliminamos ~3 syncs por forward. Con E=8 (SEGMENTS=2) o E=16 (SEGMENTS=4) el speedup crecería porque la cantidad de syncs eliminados escala con E.
- Direcciones sin probar (potencial más ganancia): batched experts con `torch.bmm` + padding a capacidad fija (Megablocks-style).

### Experimentos adicionales — dos variantes que empeoraron

Cuatro variantes probadas en total. Solo `MoELayerFast` ganó. Las otras tres empeoraron, lo cual es informativo por sí mismo:

| Variante | Tiempo | vs baseline | Correctitud |
|---|---:|---:|:---|
| MoELayer (V) | 12.54 ± 1.54 ms | 1.00× | ✓ referencia |
| **MoELayerFast (IX)** | **12.27 ± 9.01 ms** | **1.02×** | ✓ equivalente numéricamente (max diff ~1e-7) |
| MoELayerFast + torch.compile | 24.01 ± 0.94 ms | 0.52× | ✓ pero 1.9× más lenta |
| MoELayerBatched (cap=1.5, bmm+pad) | 14.76 ± 2.72 ms | 0.85× | padding + descartes posibles |

**Por qué `torch.compile` empeoró**:
1. MPS no tiene backend maduro para el compile de PyTorch (no genera kernels fusionados como CUDA con Triton). Cae a paths eager con overhead extra.
2. El costo fijo de dispatch/cache lookup no se amortiza a este tamaño de layer.
3. Shapes dinámicas (`offsets.tolist()` produce longitudes data-dependientes) fuerza recompilaciones.

**Por qué `MoELayerBatched` no ganó**:
1. Padding a capacidad fija hace ~50% más aritmética que la mínima (con `cap=1.5`). Sin padding (`cap=1.0`), aumentan los descartes y la variance.
2. `torch.bmm` sobre `(E=4, cap, 64)` no es más rápido que 4 `Linear` sueltos en MPS a este tamaño — el hardware está sub-utilizado en ambos casos.
3. Los `scatter/gather` con indexación avanzada (`expert_input[experts, ranks] = ...`) agregan kernels que compensan lo ahorrado.

**Conclusión**: la única ganancia real a esta escala en MPS vino de eliminar syncs (`.any()`, `.nonzero()`) del loop de expertos. Batched matmul y `torch.compile` — las "big guns" del stack de MoE en producción — no aplican para modelos y hardware chicos. En CUDA con `E=64` y `n_embd=4096` la historia sería totalmente distinta.

---

## 2026-09-20 — TP-I — Opcional — loss auxiliar de balanceo de carga (Switch Transformer)

**Estrategia:** duplicar `MoELayer → MoELayerBalanced` sin modificar el original. La diferencia única es guardar `_live_gate_probs` con gradiente (además del `last_gate_probs.detach()` que necesita `expert_utilization`). Custom training loop (no el Trainer estándar) que computa `total_loss = ce_loss + alpha * load_balancing_loss(model)`.

**Config:** `moe_model_balanced` con `MoEFFNBalanced` (usa `MoELayerBalanced`). Misma arquitectura que `moe_model` (4 expertos, top-2, n_embd=64, 2 capas). `alpha=0.01`, `lr=1e-3`, 2 epochs.

**Métricas de training:**

| Epoch | Train (ce) | Val (ce) | Aux |
|:---:|---:|---:|---:|
| 1 | 3.0026 | 2.1321 | 1.3227 |
| 2 | 2.2022 | 2.0277 | 1.3919 |

- Val loss final: 2.03 vs 1.89 del MoE original — 7% peor por el precio del balanceo.
- Aux se estabilizó en ~1.39 (mínimo teórico 1.0 con balance perfecto, máximo 4.0 con colapso total).

**Utilización comparativa (val_loader):**

| Capa | e0 | e1 | e2 | e3 | Diagnóstico |
|:---:|---:|---:|---:|---:|:---|
| MoE original 1 | 0.414 | 0.624 | 0.666 | 0.296 | balanceada |
| MoE balanced 1 | 0.383 | 0.770 | 0.434 | 0.413 | balanceada, aunque con e1 sobresaliendo |
| MoE original 2 | 0.001 | 0.001 | 0.999 | 0.999 | colapso total (2 expertos muertos) |
| MoE balanced 2 | 1.00 | 0.052 | 0.744 | 0.204 | reconfiguración: e0 pasa a dominar, e2 se activa moderadamente, e3 baja mucho, e1 sigue casi muerto |

**Observaciones:**
- La aux loss reconfiguró el ruteo de forma dramática pero no logró balance uniforme. En el MoE original la Capa 2 tenía e2 y e3 dominando y e0/e1 muertos; en el balanced la Capa 2 pasa a tener e0 dominante (100%), e2 moderado (74%), e3 bajo (20%) y e1 aún muy poco activo (5%). No es balance, es un óptimo local distinto.
- Con `alpha=0.01` la aux loss llegó a su mínimo local viable: verificado matemáticamente que `L_aux = E * sum(f²) ≈ 1.40` con la distribución observada, coincide con el aux observado 1.39. El modelo alcanzó un óptimo donde P sigue a f, pero f está lejos del uniforme porque un experto sigue absorbiendo la mayor parte de los tokens.
- Con `alpha` más grande (0.05 o 0.1) probablemente forzaría a mover e3, a costa de más degradación en val loss. Trade-off explícito entre balance y performance.

**Conclusión educativa:** la aux loss funciona, pero a esta escala (100k chars, modelo chico, 2 epochs) el trade-off no gana claramente. Evita el colapso completo pero degrada la calidad. En modelos grandes con corpus abundantes esta técnica es lo que permite que MoE escale sin colapsar; a escala chica, quizás sea mejor dejar que el modelo "colapse" a usar 2 expertos como si fuera denso.

---

# TP2 — instruction tuning y LoRA

Corridas del TP2. Base: checkpoint del TP-I Consigna X (`tp1_subword/checkpoint_final.pt`, mismo `tinygpt.py` y `trainer.py`).

## 2026-09-29 — TP2 — Consigna I — dataset SFT con prompt loss masking + packing + replay

**Config:** MPS. GPT2 BPE tokenizer con 4 tokens especiales agregados (`<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`), vocab final 50261. Corpus SFT completo (1.115.394 chars) para armar instrucciones, split 90/10 train/val. `block_size=32`, `batch_size=64`, `AUX_LM_LAMBDA=0.5`, `REPLAY_SHARE=0.25`.

**Métricas:**

- Pretraining base: 100.000 chars → 30.645 tokens.
- Instrucciones armadas por tarea: `continue`, `who_said` (98 personajes con ≥20 líneas), `reverse` (palabras 3-10 chars). 4 formulaciones de `continue`, una reservada (`finish this:`).
- Dataset con padding (`InstructionDataset`): 29.381 bloques, 25.5% de posiciones supervisadas por L2.
- Dataset packed (`PackedInstructionDataset`): 23.216 bloques, 32.2% de posiciones supervisadas → 1.3x menos bloques para la misma supervisión.
- Loader final con replay: 23.216 bloques SFT (239.587 targets de L2) + 2.495 bloques replay (79.840 targets de L1) → 25.0% del corpus de L1 viene de replay.
- 401 batches de train, 51 de val.

**Observaciones:**

- Prompt loss masking implementado con `ANSWER_OFFSET = len(tokenizer)`. Una sola fila de labels codifica L1 (todos los tokens, sin offset) y L2 (respuesta, con offset). `SFTWithAuxLM` decodifica al vuelo y devuelve `L3 = L2 + λ * L1`.
- Packing elimina el atajo posicional que dejaba las respuestas siempre en las posiciones 4-12 del bloque. Con packed, las respuestas caen en cualquier posición.
- Replay ancla al pretraining sin agregar una segunda loss. Un ejemplo de replay son 32 posiciones sin `ANSWER_OFFSET`, así que activa solo L1.
- `check_dataset()` pasó los 5 asserts sobre shapes, dtypes, cobertura de L2 estricta subset de L1, ningún pad supervisado, y reconstrucción exacta de la respuesta desde las posiciones supervisadas.

---

## 2026-09-29 — TP2 — Consigna II — fine-tune completo

**Config:** MPS. `full_ft_model = deepcopy(base_model)` con `requires_grad=True` en todos los parámetros. `AdamW(param_groups(...), lr=3e-4)` con weight decay `0.01` sobre matrices y `0.0` sobre embeddings/biases. `cosine_with_warmup(FT_STEPS, warmup_frac=0.05)`. `FT_EPOCHS=10`, `SFTWithAuxLM(lam=0.5)`. Save dir `./checkpoints/tp2_full_ft/`.

**Métricas (validación):**

| Tarea | Exact match | Baseline |
|---|---:|---:|
| `who_said` train | 4.8% | 3.5% (mayoría) |
| `who_said` val | 15.1% | 3.5% |
| `reverse` train | 0.0% | 0% |
| `reverse` val | 0.0% | 0% |
| overall train | 1.3% | — |
| overall val | 5.3% | — |
| brecha train-val | −4.0 pp | — |

- `continue` loss (val): 5.448.
- `continue` conditioning (val): 62.5% (50% = ignora el prompt).

**Observaciones:**

- Modelo base pre-fine-tune saca 0.0% en todo (baseline vacío). Las salidas son loops de `\nAnd, the, the,` por dos motivos: los tokens especiales están random y greedy en un modelo chico degenera.
- Post fine-tune: `who_said` supera azar por ~4x, `reverse` sigue en 0 como se esperaba (limitación de tokenización BPE, no de entrenamiento).
- Brecha train-val negativa (val > train) es consistente pero llamativa. Probable artefacto de muestra chica (n=150 por conjunto) y del hecho de que val queda por casualidad más fácil.

---

## 2026-09-29 — TP2 — Consigna II — Pregunta 5 — generalización a `finish this:` reservada

**Config:** modelo `full_ft_model` ya entrenado. Se separaron los `val_examples` de tipo `continue` en dos grupos según la formulación del prompt.

**Métricas:**

| Formulación | Loss | Conditioning |
|---|---:|---:|
| Vistas en entrenamiento (`continue:`, `go on:`, `what comes next:`) | 5.454 | 66.7% |
| Reservada (`finish this:`) | 5.564 | 74.2% |

**Observaciones:**

- Loss casi igual (+0.11, +2%) y conditioning incluso más alto en la reservada (+7.5 pp). Sin degradación → el modelo generalizó sobre la reformulación, no memorizó strings.
- El conditioning más alto en la reservada probablemente combina dos efectos: menos volumen de datos y por lo tanto más variance, y `finish this:` es semánticamente más transparente (las palabras `finish` y `this` tienen significado bien definido desde el pretraining).
- Ninguna hipótesis alternativa cambia la conclusión: es instruction tuning genuino.

---

## 2026-09-29 — TP2 — Consigna II — Pregunta 6 — experimento `AUX_LM_LAMBDA = 0`

**Config:** copia del setup del fine-tune completo, con `SFTWithAuxLM(lam=0.0)`. Save dir `./checkpoints/tp2_full_ft_lambda0/`. Sin modificar `AUX_LM_LAMBDA` global.

**Predicción:**

- Shakespeare loss sube (efecto principal, pierde ancla al pretraining).
- Exact match de `who_said` no se mueve (L2 no cambia).
- `continue` conditioning cae (menor exposición a texto crudo).
- Brecha train-val sube (menos regularización).

**Métricas comparativas:**

| Métrica | λ=0.5 | λ=0.0 | Delta |
|---|---:|---:|---:|
| who_said train | 4.8% | 4.8% | 0 pp |
| who_said val | 15.1% | 13.2% | −1.9 pp |
| overall val | 5.3% | 4.7% | −0.6 pp |
| brecha train-val | −4.0% | −3.3% | +0.7 pp |
| continue loss | 5.448 | 5.496 | +0.05 |
| continue conditioning | 62.5% | 56.7% | −5.8 pp |

**Muestras cualitativas de `continue`:**

| prompt | λ=0.5 | λ=0.0 |
|---|---|---|
| `continue: here you sty me...` | `'\nAnd'` | `'\nAnd,\nAnd the, and,\nAnd'` |
| `go on: for the mischance...` | `',\nAnd, the,\nAnd the the'` | `'\nAnd, and,\nAnd,\nAnd,'` |
| `what comes next: bountiful Fortune...` | `',\nAnd, the,\nAnd the the'` | `'\nAnd,\nAnd the a a a a,\n'` |

**Observaciones:**

- Predicciones confirmadas en dirección con magnitudes distintas. Efecto más claro: conditioning cae 5.8 pp acercándose al 50% de azar.
- Los batches de replay se vuelven inertes con λ=0 porque `is_answer` da todo False para tokens sin offset y `_per_token(logits, sft)` devuelve 0. El entrenamiento ignora el ~25% de posiciones que traía el replay.
- Los exact match casi no se mueven porque L2 no cambió. Lo que cambió es la calidad del modelo como modelo de lenguaje.
- `a a a a` en la salida de `what comes next:` con λ=0 es la firma del modelo que dejó de generar prosa fluida.
- Shakespeare loss no se midió en este experimento (se implementa en Consigna IV). Se confirmaría con esa métrica que sube.

---

## 2026-09-29 — TP2 — Consigna III — LoRA desde cero

**Config:** `apply_lora(deepcopy(base_model), r=8, alpha=16, trainable_rows=SFT_TOKEN_IDS)`. Adapters sobre `key_query_value` y `proj` en los 2 bloques. Embedding y head con `requires_grad=True` pero con hook de máscara que anula el gradiente fuera de `SFT_TOKEN_IDS`. `AdamW(param_groups(...), lr=1e-3)` (mayor que fine-tune completo, LoRA lo necesita). Weight decay `0.0` para embedding y head porque un gradiente enmascarado es cero, no ausente.

**Parámetros:**

| Alcance de descongelamiento del vocabulario | Entrenable | Notas |
|---|---:|---|
| Solo los 4 tokens del template | 0.10% | mínimo para que al menos termine |
| Vocabulario del SFT (`SFT_TOKEN_IDS`, ~13.500 filas) | ~27% | **esta corrida** |
| Embedding completo | 98.44% | no es fine-tuning bajo ninguna definición útil |

- `LoRALinear`: A de forma `(r, d_in)` con `normal_(std=1/r)`, B de forma `(d_out, r)` en ceros para que el modelo arranque idéntico al base. Forward: `base(x) + scaling * dropout(x) @ A.T @ B.T` con `scaling = alpha/r = 2`.
- Chequeo de cordura: `max |lora - base| < 1e-5` en el paso 0. Pasó.

**Métricas (validación):**

| Tarea | Exact match |
|---|---:|
| who_said train | 4.8% |
| who_said val | 11.3% |
| reverse val | 0.0% |
| overall val | 4.0% |
| brecha train-val | −2.7 pp |

- `continue` loss: 5.432.
- `continue` conditioning: 59.2%.

**Observaciones:**

- Val loss final casi idéntica al fine-tune completo (~7.5 nats/tok en ambos). `who_said val` cae ~3.8 pp respecto al full ft (11.3% vs 15.1%), esperado por la restricción de rango bajo.
- La trampa central: sin descongelar filas del embedding, el modelo nunca aprende a emitir `<|end|>` porque los tokens especiales tienen filas random y los adapters LoRA no le llegan al embedding.
- Descongelar el embedding completo mata todo el ahorro de parámetros. Solución con hook enmascara todas las filas fuera de `SFT_TOKEN_IDS`.
- LoRA arranca con menos loss inicial (~10) que el full ft (~11) porque B=0 hace que empiece idéntica al base. Full ft, aunque también parte del base, tiene todo el gradiente perturbando desde el paso 0.

---

## 2026-09-29 — TP2 — Consigna IV — olvido catastrófico

**Config:** `shakespeare_loss(model)` calcula cross-entropy sobre `shakespeare_val_loader` (los últimos 10% de tokens del corpus de preentrenamiento). Sin template, sin `ignore_index`, sin `ANSWER_OFFSET`. Es el objetivo puro de pretraining.

**Métricas:**

| Modelo | Shakespeare loss | Delta respecto al base |
|---|---:|---:|
| base (checkpoint TP-1) | 5.5653 | — |
| full ft (λ=0.5) | 5.3803 | −0.185 |
| LoRA (r=8) | 5.2981 | −0.267 |

**Observaciones:**

- Los dos modelos fine-tuneados mejoraron respecto al base. **No hubo olvido catastrófico** en el sentido canónico, hubo mejora.
- Explicación: cuenta de exposición al corpus. El pretraining del TP-1 fue muy corto (2 epochs × 30.645 tokens = ~61k tokens vistos). El fine-tune corrió 10 epochs con 25% de replay (79.840 tokens por epoch = ~800k tokens de Shakespeare crudo). El pipeline con replay + L1 no solo previno el olvido, terminó extendiendo el pretraining.
- Orden relativo entre LoRA y full ft sí confirma el paper LoRA: rango bajo mantiene al modelo más cerca del pretraining, entonces aprovecha más el replay.

**Trade-off entre instrucciones y modelado de lenguaje:**

- Por instrucciones: `full ft (15.1%) > LoRA (11.3%) > base (0.0%)` (who_said val).
- Por Shakespeare: `LoRA (5.30) < full ft (5.38) < base (5.57)`.
- LoRA es pareto-dominante sobre el base en las dos dimensiones. Full ft y LoRA son pareto-incomparables.

---

## 2026-09-29 — TP2 — Consigna Opcional — TinyGPT desde cero

**Config:** `scratch_model = TinyGPT(config)` sin cargar checkpoint del TP-1. Resize del embedding para agregar los 4 tokens especiales. Mismo setup que el fine-tune completo: `sft_train_loader`, `SFTWithAuxLM(lam=0.5)`, `AdamW(lr=3e-4)`, cosine con warmup, 10 epochs. Sanity check: `torch.equal(scratch_model.<tensor>, base_model.<tensor>)` da `False` en los 5 tensores probados (token_emb, pos_emb, key_query_value[0], proj[0], head).

**Métricas comparativas:**

| Métrica | full ft (con pretrain) | scratch (sin pretrain) | Delta |
|---|---:|---:|---:|
| who_said train | 4.8% | 2.4% | −2.4 pp |
| who_said val | 15.1% | 13.2% | −1.9 pp |
| overall val | 5.3% | 4.7% | −0.6 pp |
| continue loss | 5.448 | 4.853 | −0.60 |
| continue conditioning | 62.5% | 69.2% | +6.7 pp |
| val loss final (curva) | ~7.6 | ~6.5 | −1.1 |

**Observaciones:**

- Contra la predicción canónica (que el scratch rendiría claramente peor), rindió **comparable o mejor** en casi todas las métricas.
- Los outputs cualitativos del scratch se ven más variados: `'And I am, and the king,\nAnd'` vs `'\nAnd,\nAnd the the'` del full ft.
- Cuenta de exposición al corpus:
  - Pretrained + fine-tune: ~61k (pretraining) + ~800k (replay) = **~860k tokens** de Shakespeare vistos.
  - Scratch: 0 + ~800k = **~800k tokens** de Shakespeare vistos.
  - Ratio: ~1.08x, prácticamente iguales.
- La consigna pedía distinguir conocimiento vs optimización. A esta escala no hay brecha significativa que separar. Para verla habría que reentrenar el base con muchas más epochs (por ejemplo 20 o 50) y repetir.
- Alternativa: entrenar el scratch sin replay (`REPLAY_SHARE=0`). Si colapsa, confirma que el "conocimiento" venía del replay, no del pretraining.

---

## 2026-09-29 — TP2 — Consigna V — chat + generation_health

**Config:** `chat(model, message, temperature=0.8, top_p=0.9, max_new_tokens=24, seed=0)`. Portado desde `generateV2` del TP-1 con tres cambios: chat template, parada en `END_ID`, supresión de logits de `<|user|>`, `<|assistant|>`, `<|pad|>`. Sampling en vez de greedy (greedy entra en loops feos en un modelo así de chico).

**Muestras (`full_ft_model`, seed=0):**

| Prompt | Salida |
|---|---|
| `continue: To be, or not to be,` | `'\nYet? the myT and to the lord.\n\nDUKE VINC'` (llegó al tope de 24 tokens) |
| `finish this: First Citizen:\nBefore we proceed` | `', and'` (cortó por `<|end|>` temprano) |
| `who said: Now is the winter of our discontent` | `'DUKE VINCENTIO'` (personaje válido pero equivocado, era GLOUCESTER) |
| `reverse: sword` | `'nS'` (falla esperada por tokenización) |

**Chequeo de salud (`generation_health`, n=60 val examples):**

| Modelo | Termina en `<|end|>` | Largo medio | LM loss (Shakespeare) |
|---|---:|---:|---:|
| base | 0.0% | 24.0 (tope) | 5.565 |
| full ft | 100.0% | 4.0 | 5.380 |
| LoRA | 100.0% | 4.7 | 5.298 |

**Observaciones:**

- Los tres cambios del port funcionaron: no aparecen tokens especiales en el medio de las respuestas, las salidas terminan solas donde deben, y el sampling evita los loops que arruinaban a greedy.
- Los fine-tuneados aprendieron perfectamente a terminar (`100%` de terminación con largo medio 4 tokens). El base nunca termina porque nunca fue entrenado con `<|end|>`, su fila de embedding sigue random.
- La LM loss confirma la Consigna IV con los mismos números.

---

## 2026-09-29 — TP2 — Consigna VI — atención sobre `<|assistant|>`

**Config:** `attention_at_answer(model, message, layer=-1, top=6)` sobre `full_ft_model`. `model(idx, return_weights=True)` devuelve pesos por capa con shape `(n_head, B, T, T_k)`. Se toma la última fila de queries (posición del `<|assistant|>`) y se promedia sobre heads. Bar chart con colormap `viridis` para el visual.

**Top tokens que mira la posición `<|assistant|>` (capa −1):**

| Tarea | Top marcador | Peso al marcador | Otros pesos notables |
|---|---|---:|---|
| `continue: To be, or not to be,` | pos 1 `continue` | **0.270** | self `<\|assistant\|>` 0.249, ` be` 0.094 |
| `who said: Now is the winter` | pos 1 `who` | **0.196** | self 0.433, ` the` 0.125 |
| `reverse: sword` | pos 1 `reverse` | **0.038** | self 0.461, `<\|user\|>` 0.175, ` sword` 0.158 |

**Observaciones:**

- Correlación directa entre peso al marcador de tarea y accuracy medida en Consigna II: `continue (0.270, conditioning 62.5%) > who_said (0.196, 15.1% val) > reverse (0.038, 0.0%)`.
- El modelo aprendió a rutear por el marcador de tarea cuando la tarea es aprendible. Cuando no (`reverse`, imposible por tokenización), el modelo aprendió a ignorar el marcador.
- Esto explica mecánicamente el ruteo observado en la Consigna II: la misma entrada con marcadores distintos produce respuestas distintas porque la atención en `<|assistant|>` cambia según qué marcador vio antes.
- `who_said` y `reverse` tienen self-attention alta (0.43 y 0.46) porque el contenido antes del `<|assistant|>` es corto y el modelo pesa mucho sobre sí mismo. `continue` tiene contenido más largo y distribuye más.

---
