# TPs PNL-II (CEIA)

Repositorio de entregas de los TPs de NLP-II / LLMs y GenIA de la CEIA (FIUBA).

## Estructura

- `TP-I/` — TinyGPT: pretraining, decodificación y Mixture of Experts.
- `PRUEBAS.md` — historial de pruebas y métricas por consigna.

## Cómo correr TP-I

```bash
cd TP-I
python -m venv .venv && source .venv/bin/activate
pip install torch matplotlib numpy tqdm httpx tiktoken jupyter transformers
# opcional, sólo CUDA:
# pip install bitsandbytes
jupyter notebook TP1_TinyGPT_es.ipynb
```

Notas:

- `TP1_TinyGPT_es.ipynb` importa `tinygpt.py` y `trainer.py` desde la misma carpeta.
- Entrena 4 modelos en el mismo kernel y guarda checkpoints en `./checkpoints/tp1_*`.
- Bajar `N_CHARS` (default `100_000`) si el entrenamiento es lento.
- macOS/MPS: usar `NUM_WORKERS=0`. Adam-8bit, `torch.compile` y autocast bfloat16 sólo se activan con CUDA.
- Tras reiniciar el kernel, correr `resume_all()` para restaurar modelos ya entrenados sin reentrenar.
