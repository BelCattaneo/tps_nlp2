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
