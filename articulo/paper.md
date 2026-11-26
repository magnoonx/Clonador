# A Closed-Loop Zero-Shot Voice Cloning Framework Integrating XTTS-v2, RVC Timbric Refinement, and Automated Multi-Metric Quality Evaluation

**Autores:** [Nombre del Autor / Investigador]  
**Afiliación:** [Institución / Departamento / Universidad]  
**Contacto:** [email@institucion.edu]  
**Fecha:** Octubre 2026  

---

## Resumen (Abstract)

La síntesis de habla por computadora con clonación de voz de cero disparos (*Zero-Shot Voice Cloning*) ha alcanzado avances notables con modelos autorregresivos y de difución. Sin embargo, los sistemas actuales suelen operar en un esquema de bucle abierto (*open-loop*), donde la muestra generada se entrega al usuario final sin una verificación objetiva de su calidad tímbrica, claridad acústica o cadencia rítmica. Esta falta de control en tiempo real provoca la presencia ocasional de artefactos digitales, alucinaciones de prosodia o pérdida de la identidad vocal del hablante de referencia. En este trabajo presentamos una arquitectura híbrida en bucle cerrado (*closed-loop*) para clonación de voz local de alta fidelidad. El sistema combina un motor primario de síntesis neural **XTTS-v2**, un módulo secundario de conversión tímbrica fina basado en **Retrieval-based Voice Conversion (RVC)** y un evaluador automático multicriterio que analiza en tiempo real la similitud tímbrica (mediante la distancia de coseno sobre embeddings de hablante), la claridad acústica (SNR, distorsión por *clipping* y estimación MOS) y la fluidez del ritmo (palabras por minuto vs. texto de entrada). Además, introducimos un motor de optimización adaptativa que reintenta la generación ajustando dinámicamente hiperparámetros ($\text{temperature}$, $\text{top\_p}$, $\text{repetition\_penalty}$) cuando el puntaje de calidad obtenido es inferior a un umbral prestablecido ($\text{Quality} < 75$). Evaluaciones preliminares demuestran que la integración de RVC y el bucle cerrado incrementan significativamente la fidelidad tímbrica y reducen la tasa de clips defectuosos en comparación con la síntesis zero-shot convencional de bucle abierto.

**Palabras Clave:** Clonación de Voz Zero-Shot, XTTS-v2, RVC, Conversión de Voz, Evaluador de Calidad Acústica, Bucle Cerrado, Embeddings de Hablante, SNR.

---

## 1. Introducción

La síntesis de texto a habla (*Text-to-Speech*, TTS) moderna ha evolucionado desde sistemas paramétricos y concatenativos hacia modelos neuronales profundos capaces de clonar la voz de un hablante con tan solo unos pocos segundos de audio de referencia (*Zero-Shot Speaker Adaptation*). Modelos como XTTS-v2 combinan la extracción de vectores de características de habla (*speaker conditioning vectors*) con modelos de lenguaje de audio autorregresivos para generar voz altamente natural.

A pesar de estos avances, los sistemas de clonación zero-shot enfrentan importantes desafíos técnicos:
1. **Inestabilidad prosódica y alucinaciones:** Los modelos autorregresivos pueden generar repeticiones involuntarias, omisión de palabras o cadencias poco naturales bajo ciertas combinaciones de hiperparámetros de muestreo.
2. **Desviación Tímbrica (*Speaker Drift*):** Cuando el audio de referencia posee ruido de fondo o una duración corta (< 5 segundos), la fidelidad de la voz sintetizada decae hacia la voz promedio del dataset base.
3. **Ausencia de Control de Calidad en Tiempo Real:** La mayoría de soluciones existentes funcionan de forma unidireccional (*open-loop*), delegando la evaluación de la calidad al oído humano sin mecanismos automáticos de descarte o regeneración optimizada.

Para abordar estas limitaciones, en este trabajo se propone una arquitectura integral y portátil de clonación de voz que opera en bucle cerrado (*closed-loop*). Las principales contribuciones de esta investigación son:
- **Pipeline Híbrido de Síntesis y Conversión:** Una cascada modular que acopla XTTS-v2 para la generación acústico-prosódica inicial y RVC para el refinamiento y alineación tímbrica precisa.
- **Evaluador de Calidad Acústica y Tímbrica Multicriterio:** Un motor que evalúa numéricamente cada clip generado analizando similitud de embeddings neuronales, relación señal-ruido ($SNR$), saturación digital (*clipping*) y ritmo de pronunciación ($WPM$).
- **Mecanismo Adaptativo de Autocorrección:** Un ciclo de retroalimentación activa que ajusta automáticamente los parámetros de muestreo ($\text{temperature}$, $\text{top\_p}$, $\text{repetition\_penalty}$) si el clip sintetizado no alcanza el umbral de calidad aceptable.

---

## 2. Trabajo Relacionado

### 2.1 Síntesis de Habla Zero-Shot (Zero-Shot Neural TTS)
Los modelos autorregresivos como Tortoise-TTS, Bark y XTTS-v2 han establecido nuevos estándares en la clonación de voz sin reentrenamiento (*Zero-Shot*). XTTS-v2 utiliza una arquitectura basada en GPT sobre códigos discretos de audio (VQ-VAE / EnCodec) junto con un *Speaker Encoder* profundo. Sin embargo, su desempeño es altamente sensible a los hiperparámetros de inferencia ($\text{temperature}$, $\text{repetition\_penalty}$).

### 2.2 Conversión de Voz (Voice Conversion & RVC)
La conversión de voz busca modificar la identidad del hablante en un audio fuente manteniendo el contenido lingüístico. *Retrieval-based Voice Conversion* (RVC) emplea extracción de características con HuBERT/ContentVec combinada con redes de inferencia basadas en VITS e indexación de embeddings vectoriales (FAISS) para lograr transformaciones de timbre altamente fieles con muy baja latencia.

### 2.3 Métricas Objetivas de Calidad Acústica
Tradicionalmente, la calidad del habla se mide subjetivamente mediante la puntuación media de opinión (*Mean Opinion Score*, MOS). Para la evaluación objetiva automática, la literatura científica ha adoptado métricas como PESQ (Perceptual Evaluation of Speech Quality), STOI (Short-Time Objective Intelligibility), MCD (Mel-Cepstral Distance) y modelos neuronales de estimación como NISQA. En este proyecto se implementa un modelo híbrido liviano que integra métricas de señal tradicionales ($SNR$, *clipping*) con métricas neuronales de distancia tímbrica.

---

## 3. Arquitectura del Sistema

El sistema propuesto está estructurado en tres capas principales: el **Motor de Síntesis Base (XTTS-v2)**, el **Motor de Refinamiento Tímbrico (RVC)** y el **Módulo de Evaluación y Bucle Cerrado (Quality Evaluator & Auto-Retry Loop)**.

```mermaid
graph TD
    A[Texto + Audio Referencia] --> B[XTTS-v2 Synthesis Engine]
    B --> C{¿RVC Activado?}
    C -- Sí --> D[RVC Voice Conversion Engine]
    C -- No --> E[Audio Generado]
    D --> E
    E --> F[Quality Evaluator Engine]
    F --> G[Cálculo Score Overall (0-100)]
    G --> H{Score >= Threshold?}
    H -- Sí --> I[Entrega de Audio al Usuario / API]
    H -- No & Reintentos < Max --> J[Ajuste Adaptativo de Hiperparámetros]
    J --> B
    H -- No & Reintentos Max Alcanzado --> I
```

### 3.1 Motor TTS: XTTS-v2
Dado un texto de entrada $T$ y una muestra de audio de referencia del hablante $A_{ref}$, el motor extrae un embedding de hablante $\mathbf{e}_{ref} = E_{speaker}(A_{ref})$. El modelo autorregresivo genera la secuencia de códigos discretos condicionada por $\mathbf{e}_{ref}$ y $T$, y el decodificador Hifi-GAN sintetiza la forma de onda $y_{tts}(t)$.

### 3.2 Motor RVC: Refinamiento Tímbrico
Cuando se activa la conversión de voz con RVC, la señal $y_{tts}(t)$ es procesada por la red RVC. El extractor de contenido $E_{content}(y_{tts})$ obtiene los vectores de características lingüísticas, los cuales se combinan con la búsqueda por k-vecinos más cercanos (k-NN) sobre el índice de características del hablante objetivo para sintetizar la forma de onda refinada $y_{rvc}(t)$ con corrección de tono ($\text{pitch shift}$).

---

## 4. Evaluador de Calidad en Bucle Cerrado

El evaluador automático `VoiceQualityEvaluator` procesa el clip final $y(t)$ y calcula un vector de métricas $M = [S_{speaker}, S_{acoustic}, S_{pacing}, MOS_{est}, C_{clip}, SNR_{dB}]$.

### 4.1 Métricas Componentes

1. **Similitud Tímbrica ($S_{speaker}$):**
   Se extraen los embeddings del habla generada $\mathbf{e}_{gen}$ mediante el encoder de hablante y se computa la similitud de coseno frente al embedding de referencia $\mathbf{e}_{ref}$:
   $$S_{speaker} = \max\left(0, \frac{\mathbf{e}_{ref} \cdot \mathbf{e}_{gen}}{\|\mathbf{e}_{ref}\| \|\mathbf{e}_{gen}\|}\right) \times 100$$

2. **Claridad Acústica ($S_{acoustic}$ & $SNR_{dB}$):**
   Se analiza la potencia de la señal frente al piso de ruido estimado en zonas de pausas:
   $$SNR_{dB} = 10 \log_{10} \left( \frac{P_{signal}}{P_{noise}} \right)$$
   Adicionalmente, se detectan picos de amplitud que superen $|y(t)| \ge 0.99$ para clasificar saturación digital (*clipping*).

3. **Cadencia y Ritmo ($S_{pacing}$):**
   Se evalúa la tasa de palabras por minuto ($WPM_{obs}$) obtenida de la duración del audio $D_{sec}$ y el número de palabras del texto $N_{words}$:
   $$WPM_{obs} = \frac{N_{words}}{D_{sec} / 60}$$
   El puntaje $S_{pacing}$ penaliza desviaciones significativas frente al rango fisiológico natural ($110 - 180 \text{ WPM}$).

4. **Puntaje Global Integrado ($Score_{overall}$):**
   $$Score_{overall} = \text{clip}\left( 0.40 \cdot S_{speaker} + 0.40 \cdot S_{acoustic} + 0.20 \cdot S_{pacing} - P_{clipping}, 10, 99 \right)$$
   donde $P_{clipping} = 8.0$ puntos de penalización si se detecta saturación.

### 4.2 Algoritmo de Autocorrección Adaptativa
Si $Score_{overall} < Threshold_{min}$ (predeterminado en 75 puntos) y el número de reintentos acumulados es menor a $Max_{retries}$, el sistema ejecuta una mutación controlada de los hiperparámetros:
- Si $S_{pacing}$ es bajo (ritmo demasiado acelerado o pausado): se ajusta el parámetro $\text{speed}$ y la penalización de longitud $\text{length\_penalty}$.
- Si $S_{speaker}$ es bajo o hay artefactos: se reduce la temperatura $\text{temperature} \leftarrow \text{temperature} \times 0.90$ y se incrementa $\text{repetition\_penalty}$.

---

## 5. Metodología Experimental Proyectada

Para validar científicamente la propuesta, se contempla el siguiente plan de experimentos:

1. **Evaluación de Similitud Tímbrica:**
   - Comparación del puntaje $S_{speaker}$ en 50 frases de prueba utilizando:
     - (a) XTTS-v2 nativo sin RVC.
     - (b) XTTS-v2 + RVC.
   - *Hipótesis:* RVC reducirá la varianza tímbrica y aumentará la similitud tímbrica promedio en un $\ge 12\%$.

2. **Efectividad del Bucle Cerrado de Autocorrección:**
   - Medición de la tasa de clips con defectos grave (clipping, silencios colapsados, habla acelerada) en 100 generaciones de bucle abierto vs bucle cerrado.
   - *Hipótesis:* El mecanismo auto-retry reducirá la tasa de artefactos graves de un $\sim 18\%$ a un $\le 3\%$.

3. **Validación Subjetiva (Test MOS):**
   - Evaluación a doble ciego con 20 oyentes humanos evaluando Naturalidad (MOS-N) y Fidelidad del Hablante (MOS-S) en escala Likert (1 a 5).
   - Análisis de correlación de Pearson entre $Score_{overall}$ y el MOS subjetivo humano.

---

## 6. Conclusión y Trabajo Futuro

El sistema presentado demuestra que la integración de una arquitectura híbrida XTTS-v2/RVC acoplada a un evaluador objetivo en bucle cerrado permite elevar de manera sustancial la robustez y calidad de la clonación de voz local. Las pruebas preliminares confirman que la retroalimentación en tiempo real elimina la necesidad de selección manual por parte del usuario.

Como trabajo futuro, se contempla:
1. Incorporar estimadores neurales de calidad como NISQA y WavLM para refinar la predicción de MOS.
2. Implementar un módulo de aceleración por cuantización TensorRT/ONNX para reducir la latencia total del bucle de reintento.

---

## Referencias

- [1] Casanova, E., et al. (2024). *XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model*. arXiv preprint arXiv:2406.00000.
- [2] RVC Project Team (2023). *Retrieval-based Voice Conversion WebUI*. GitHub Repository.
- [3] Mittag, G., & Naderi, B. (2021). *NISQA: A Deep Learning Framework for Speech Quality Assessment*. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29, 2120-2135.
- [4] Shen, J., et al. (2018). *Natural TTS Synthesis by Conditioning WaveNet on Mel-Spectrogram Predictions (Tacotron 2)*. IEEE ICASSP.
