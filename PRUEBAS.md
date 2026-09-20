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
- Epoch 1: train 2.1788 · val 2.0508.
- Epoch 2: train 2.0885 · val 1.9845.
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
| 50 | 0.101 | 0.117 | 1.16× |
| 100 | 0.109 | 0.120 | 1.09× |
| 200 | 0.172 | 0.169 | 0.98× |
| 400 | 0.238 | 0.240 | 1.01× |

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
- Epoch 1: train 2.0436 · val 1.9321. Throughput: 40 it/s (train) vs 149 it/s del denso — **~3.7× más lento por paso**.
- Epoch 2: train 1.9657 · val 1.8580. Throughput: 70 it/s (train) — mejora tras JIT warm.
- Val loss final: **denso 1.9845 vs MoE 1.8580** — mejora de 0.126.

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
| 0 | 0.609 | 0.469 | 0.537 | 0.386 | Sí, entre 0.39 y 0.61 |
| 1 | 0.00005 | 0.999 | 0.016 | 0.985 | No — colapso total a e1 y e3 |

**Observaciones:**
- Capa 0 está balanceada dentro de un rango razonable (todos entre 0.39 y 0.61, ideal 0.5).
- **Capa 1 colapsó totalmente**: e1 y e3 reciben ~100% de los tokens cada uno (o sea, todos los tokens siempre se rutean a esos dos). e0 y e2 están muertos (<2%).
- El colapso se refuerza solo: los expertos no elegidos no reciben gradiente, la gate aprende a bajar aún más su score, ciclo winner-take-all clásico.
- Sin loss auxiliar de balanceo (Switch Transformers, ec. 4), nada contrarresta el colapso.
- La Capa 1 (cerca de la loss) colapsó más que la Capa 0 — argumento probable: gradientes más fuertes cerca de la loss aceleran el winner-take-all.
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
| Denso | 106.048 | 1.9845 | 226 |
| MoE (VI) | 305.088 | 1.8580 | 70 |
| DeepSeekMoE | 339.264 | **1.8363** | 38 |

- DS beat MoE por 0.022 en val loss. Reproduce la dirección del paper pero no la magnitud.
- Costo: ~40% más de tiempo por paso vs MoE.

**Utilización de expertos (DS, val_loader):**

| Capa | Fracciones (ideal k/E = 0.5) | Diagnóstico |
|:---:|---|:---|
| 0 | [0.50, 0.56, 0.69, 0.43, 0.45, 0.57, 0.29, 0.51] | Balanceada — todos entre 0.29 y 0.69, ninguno muerto. |
| 1 | [0.83, 0.003, 0.87, 0.10, 1.00, 0.028, 0.17, 1.00] | Colapso parcial — 4 dominan (0.83–1.00), 4 casi muertos (0.003–0.17). Ninguno *totalmente* muerto (a diferencia del MoE original que dejaba e0 en 0.005%). |

- Comparado con el MoE (Capa 1 con 2 fully active + 2 dead), el DS tiene 4 fully active + 4 casi-dead. Proporcionalmente el desbalance es similar, pero absolutamente hay más flexibilidad (los "perdedores" reciben algo de gradiente y podrían recuperarse).

**Magnitud shared vs routed (Block 0):**
- Norma L2 promedio shared: 7.08.
- Norma L2 promedio routed: 18.99.
- Ratio shared/routed: **0.37**.
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
| MoELayer (V) | 12.94 ± 2.96 ms | alta |
| MoELayerFast (IX) | 10.45 ± 0.44 ms | 6.7× menor |
| **Speedup** | **1.24×** | — |

- Correctitud verificada con `test_moe_equivalence`: max abs diff ~1e-7 (dentro de tolerancia float32).

**Observaciones:**
- Ganancia modesta (1.24×) porque E=4 es chico — sólo eliminamos ~3 syncs por forward.
- Lo más informativo: variance 6.7× menor. La baseline tenía picos por syncs impredecibles; la fast es consistente.
- Con E=8 (SEGMENTS=2) o E=16 (SEGMENTS=4) el speedup crecería porque la cantidad de syncs eliminados escala con E.
- Direcciones sin probar (potencial más ganancia): batched experts con `torch.bmm` + padding a capacidad fija (Megablocks-style).

### Experimentos adicionales — dos variantes que empeoraron

Cuatro variantes probadas en total. Solo `MoELayerFast` ganó. Las otras tres empeoraron, lo cual es informativo por sí mismo:

| Variante | Tiempo | vs baseline | Correctitud |
|---|---:|---:|:---|
| MoELayer (V) | 12.94 ± 2.96 ms | 1.00× | ✓ referencia |
| **MoELayerFast (IX)** | **10.45 ± 0.44 ms** | **1.24×** | ✓ equivalente numéricamente (max diff ~1e-7) |
| MoELayerFast + torch.compile | 24.45 ± 1.74 ms | 0.53× | ✓ pero 1.9× más lenta |
| MoELayerBatched (cap=1.5, bmm+pad) | 13.40 ± 0.39 ms | 0.97× | padding + descartes posibles |
| MoELayerBatched (cap=1.0, bmm+pad) | 14.16 ± 4.54 ms | 0.91× | descartes más probables → mayor variance |

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
| 1 | 3.0028 | 2.1328 | 1.3128 |
| 2 | 2.1929 | 2.0329 | 1.3715 |

- Val loss final: 2.03 vs 1.86 del MoE original — 9% peor por el precio del balanceo.
- Aux se estabilizó en ~1.37 (mínimo teórico 1.0 con balance perfecto, máximo 4.0 con colapso total).

**Utilización comparativa (val_loader):**

| Capa | e0 | e1 | e2 | e3 | Diagnóstico |
|:---:|---:|---:|---:|---:|:---|
| MoE original 0 | 0.609 | 0.469 | 0.537 | 0.386 | balanceada |
| MoE balanced 0 | 0.435 | 0.425 | 0.537 | 0.603 | balanceada (más apretada al ideal 0.5) |
| MoE original 1 | 0.00005 | 0.999 | 0.016 | 0.985 | colapso total (2 expertos muertos) |
| MoE balanced 1 | 0.221 | 0.503 | 0.276 | 0.9999 | mejora parcial: e0/e2 resucitados, e3 sigue dominante |

**Observaciones:**
- La aux loss evitó las muertes de expertos (e0 pasó de 0.005% a 22%, e2 de 1.6% a 28%) pero no logró desplazar a e3 (0.985 → 0.9999). Mejora parcial, no balance completo.
- Con `alpha=0.01` la aux loss llegó a su mínimo local viable: verificado matemáticamente que `L_aux = E * sum(f²) ≈ 1.38` con la distribución observada, coincide con el aux observado 1.37. El modelo alcanzó un óptimo donde P sigue a f, pero f está lejos del uniforme porque e3 es estructuralmente "necesario" para la tarea principal.
- Con `alpha` más grande (0.05 o 0.1) probablemente forzaría a mover e3, a costa de más degradación en val loss. Trade-off explícito entre balance y performance.

**Conclusión educativa:** la aux loss funciona, pero a esta escala (100k chars, modelo chico, 2 epochs) el trade-off no gana claramente. Evita el colapso completo pero degrada la calidad. En modelos grandes con corpus abundantes esta técnica es lo que permite que MoE escale sin colapsar; a escala chica, quizás sea mejor dejar que el modelo "colapse" a usar 2 expertos como si fuera denso.

---
