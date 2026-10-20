# Detector de Arritmias Cardíacas — MIT-BIH
**Biomedical Engineering Project | Universidad EIA**

Pipeline de clasificación de latidos ECG en 5 clases AAMI (N, S, V, F, Q) usando una CNN 1D con bloques residuales, evaluada con el protocolo **inter-paciente** de De Chazal et al. (2004).

> Proyecto académico. No es un dispositivo médico ni sirve para diagnóstico.

---

## Estructura del proyecto

```
arritmias/
├── ingesta_mitdb.py       # Paso 1: extrae latidos y crea splits por paciente
├── segmentacion_ecg.py    # Paso 2: filtra y segmenta ventanas centradas en el pico R
├── modelo_cnn.py          # Paso 3: entrena la CNN 1D y evalúa en test
├── inferencia.py          # Paso 4: clasifica un registro MIT-BIH o una señal propia
├── requirements.txt
├── tests/
│   ├── make_synthetic_mitdb.py   # ECG sintético en formato MIT-BIH (solo para pruebas)
│   └── test_pipeline.py          # prueba de humo de punta a punta (pytest)
├── data/
│   ├── mitdb/             # Archivos MIT-BIH (.hea, .dat, .atr)
│   ├── beats.parquet      # Salida paso 1
│   ├── splits.json        # Salida paso 1
│   └── windows.npz        # Salida paso 2
└── output/
    ├── modelo_arritmias.pt
    ├── metricas_test.json     # métricas por clase + matriz de confusión
    └── matriz_confusion.png
```

---

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Pipeline completo

```bash
# 1) Ingesta: latidos + splits (De Chazal por defecto)
python ingesta_mitdb.py --records data/mitdb/ --outdir data/

# 2) Segmentación: ventanas de 216 muestras (600 ms a 360 Hz), filtradas y normalizadas
python segmentacion_ecg.py --records data/mitdb/ --beats data/beats.parquet --outdir data/

# 3) Entrenamiento y evaluación en test
python modelo_cnn.py --windows data/windows.npz --splits data/splits.json --outdir output/ --epochs 30

# 4) Inferencia
python inferencia.py --model output/modelo_arritmias.pt --record data/mitdb/100
python inferencia.py --model output/modelo_arritmias.pt --signal mi_ecg.csv --fs 250
```

Para verificar que todo el código funciona sin datos reales: `pytest -q`.

---

## Decisiones técnicas

**Split inter-paciente (De Chazal).** DS1 (22 registros) se divide en train/val y DS2 (22 registros) es el test. Se excluyen los 4 registros con marcapasos (102, 104, 107, 217). Así los resultados son comparables con la literatura y no hay fuga entre pacientes. `--split_mode random` queda como alternativa, pero con 48 registros el test puede quedar sin clases F o S.

**Canal MLII por nombre.** El canal 0 no siempre es MLII en MIT-BIH (p. ej. el registro 114 lo trae en el canal 1), por eso `segmentacion_ecg.py` lo busca por nombre.

**Selección del mejor modelo por F1 macro de validación, no por accuracy.** Con N ≫ S, V, F, Q, un modelo que predice siempre "N" ya tiene accuracy muy alta.

**Desbalance: una sola técnica a la vez.** `--balance sampler` (default), `loss` o `none`. Usar `WeightedRandomSampler` y pesos en la loss simultáneamente sobre-compensa a las clases raras.

**Mismo preprocesamiento en entrenamiento e inferencia.** `inferencia.py` reutiliza `bandpass`, `normalize` y `cut_windows` de `segmentacion_ecg.py` para evitar *train/serve skew*.

---

## Clases AAMI

| Clase | Descripción | Símbolos MIT-BIH |
|-------|-------------|------------------|
| N | Normal + BBB + escapes | N, L, R, e, j |
| S | Supraventricular ectópico | A, a, J, S |
| V | Ventricular ectópico | V, E |
| F | Fusión ventricular/normal | F |
| Q | Paced / desconocido | /, f, Q |

---

## Arquitectura — ArrhythmiaNet

```
Input (1, 216)
  └─ Stem Conv1D(1→32, k=15) + BN + ReLU + MaxPool → (32, 108)
       └─ ResBlock(32, k=7)
            └─ Down(32→64) + MaxPool → (64, 54)
                 └─ ResBlock(64, k=5)
                      └─ Down(64→128) + MaxPool → (128, 27)
                           └─ ResBlock(128, k=3)
                                └─ AdaptiveAvgPool → (128, 1)
                                     └─ FC(128→64→5)
```

---

## Resultados

**Pendiente: ejecutar el paso 3 con MIT-BIH real y pegar aquí la tabla de `output/metricas_test.json`.**

| Clase | Precision | Recall | F1 | Soporte |
|-------|-----------|--------|----|---------|
| N | _por medir_ | | | |
| S | _por medir_ | | | |
| V | _por medir_ | | | |
| F | _por medir_ | | | |
| Q | _por medir_ | | | |

Como referencia (no son resultados de este proyecto), en la literatura con el protocolo De Chazal las clases N y V suelen superar F1 ≈ 0.9, mientras que S y F son mucho más difíciles. Reporta solo lo que midas.

---

## Limitaciones conocidas
- Un solo canal (MLII) y ventanas de 600 ms: no usa el intervalo RR previo/posterior, información muy útil para distinguir latidos S.
- `inferencia.py` usa un detector de picos R sencillo; para señales ruidosas hay que validarlo o cambiarlo.
- MIT-BIH tiene solo 47 pacientes: el modelo no está validado en otros equipos, poblaciones ni condiciones de ruido.

## Próximos pasos sugeridos
- [ ] Añadir características RR (intervalo previo, posterior, promedio local) como segunda entrada
- [ ] Aumentación de datos (ruido gaussiano, time-warping)
- [ ] Grad-CAM para interpretabilidad
- [ ] Exportar a ONNX para deployment en Jetson Nano
- [ ] Validar con datos del lector propio
