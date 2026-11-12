# Radar de narrativas

**Desafío QualyData · Hackathon Argentina Space 2026 (Salta, 11 al 13 de noviembre)**

Un pipeline que, sobre cientos de miles de comentarios, responde dos preguntas: **qué narrativas circulan** (quién las apoya, quién las rechaza y desde cuándo) y **qué parte de la conversación está armada** (qué cuentas actúan coordinadas y en qué operación). Corre con un comando, en una notebook sin GPU, y usa IA solo sobre muestras: el costo no crece con el volumen.

En el set de práctica (200.000 comentarios) saca **53,8 de 55 puntos automáticos** con **US$ 0,012 de IA en total** (seis millonésimas de dólar por comentario), en poco más de un minuto con la caché de embeddings y unos 6 minutos la primera vez. Sin IA saca 51,4. La línea base del desafío saca 16,8.

## Cómo correrlo

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.12 solo), unos 2 GB de disco, sin GPU.

```bash
make instalar
```

```bash
make modelos
```

```bash
make final
```

`make final` lee `datos/final/` y deja el resultado en `entrega/`. Es lo mismo que:

```bash
uv run radar --datos datos/final --entrega entrega
```

**IA opcional.** Con una clave de Anthropic en un archivo `.env` (ignorado por git) el pipeline usa Claude para nombrar narrativas y etiquetar muestras de postura:

```bash
echo "ANTHROPIC_API_KEY=..." > .env
```

Sin clave, o sin internet, corre igual: lo detecta al arrancar y sigue sin IA. `--sin-ia` lo fuerza y `--modelo-ia` elige el modelo (por defecto `claude-haiku-5-5`).

Otros comandos útiles: `make datos` baja el material del desafío, `make practica` corre y puntúa la práctica, `make sucio` prueba con datos corruptos y `make rodajas` prueba la generalización por tema.

### Reproducibilidad

- Todos los pasos aleatorios usan semillas fijas (k-means, Leiden, muestreo) y las versiones exactas están en `uv.lock`.
- Cada respuesta de la IA queda registrada en `registro_ia/<set>.jsonl`, con el pedido, la respuesta, los tokens y el costo. Si se vuelve a correr sobre los mismos datos, el pipeline reutiliza esas respuestas en vez de llamar a la API: el resultado es idéntico aunque no haya clave. Lo verificamos con tres corridas (respuestas nuevas, repetida y con una clave inválida): los tres archivos de la entrega salieron idénticos byte a byte.
- `--ia-nueva` ignora el registro y vuelve a pedir las respuestas; `--sin-ia` corre sin IA. `costo.json` separa las llamadas nuevas de las reutilizadas.

## Qué produce

En `entrega/`:

| Archivo | Contenido |
|---|---|
| `comentarios.csv` | Narrativa y postura (`apoya` / `rechaza`) de cada comentario; vacío si es ruido |
| `narrativas.csv` | Afirmación, frases clave y fecha de origen de cada narrativa |
| `cuentas.csv` | Operación coordinada de cada cuenta, vacío si es genuina |
| `costo.json` | Tiempo por etapa, tokens y dólares de IA, hardware y almacenamiento |
| `tablero.html` | Tablero autocontenido: qué circula, desde cuándo y quién lo empuja. Abre sin internet |

## Cómo funciona

```
CSV.gz ─► ingesta (DuckDB) ─► Parquet ─┬─► narrativas ─► nombres (IA) ─► postura (IA en muestras) ─┐
                                       └─► coordinación ──────────────────────────────────────────┴─► entrega/
```

1. **Ingesta.** DuckDB lee todo como texto y lo limpia explícitamente: ids repetidos, fechas mal formadas o en otros formatos, respuestas a comentarios inexistentes, cuentas o publicaciones faltantes, `me_gusta` no numéricos, líneas CSV rotas, columnas de más, de menos o en otro orden, archivos sin comprimir o en otra codificación. Cada problema se cuenta en `trabajo/<set>/reporte.json` en vez de cortar la corrida.
2. **Embeddings locales.** Cada texto único (después de normalizar abreviaturas y letras repetidas) pasa por [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), que corre en CPU. Los vectores quedan en caché por texto, así que repetir una frase no cuesta nada.
3. **Narrativas.** No hay lista previa ni cantidad fija:
   - k-means parte los textos en cientos de micro-grupos muy puros;
   - un micro-grupo cuyos comentarios casi nunca participan de hilos de respuestas es ruido (charla fuera de tema que nadie contesta), igual que los comentarios muy cortos fuera de hilos;
   - los micro-grupos se unen en narrativas cuando sus comentarios se responden entre sí **más de lo esperable por azar** (una respuesta discute dentro de la narrativa de su padre, aunque la rechace) o cuando dicen casi lo mismo. Leiden encuentra los grupos. Medir contra el azar ("lift") hace que funcione igual con 60.000 o con 200.000 comentarios;
   - cada hilo vota una sola narrativa y las respuestas cortas ("Tal cual") heredan la de su padre;
   - el origen de cada narrativa es la primera ráfaga de sus comentarios más típicos.
4. **Nombres (IA, una llamada).** Claude ve una docena de comentarios representativos de cada narrativa y devuelve su afirmación en una oración, frases clave, si dos grupos son la misma narrativa y si un grupo no es narrativa.
5. **Postura.** Por narrativa, Claude etiqueta unas 60 muestras elegidas para cubrir todos los argumentos (30 diversas y, si el reparto sale desparejo, 30 donde el clasificador duda) y una regresión logística sobre los embeddings etiqueta el resto en milisegundos. Las respuestas cortas heredan la postura de su padre; las dudosas toman la contraria, porque la mayoría de las respuestas discute.
6. **Coordinación.** Una ráfaga no prueba nada; lo que delata a una operación es la repetición: las mismas cuentas, juntas, una y otra vez.
   - Cada cuenta es un vector sobre "eventos" (publicación × ventana de 15 y de 60 minutos) pesados por rareza: coincidir en un momento tranquilo pesa mucho más que coincidir en uno viral, y las ventanas multitudinarias se ignoran.
   - Se suman hashtags raros compartidos y respuestas mutuas repetidas. Las frases repetidas solas no sirven: la gente genuina se repite muchísimo.
   - Leiden encuentra los grupos, se descartan las cuentas con menos de dos vínculos internos y solo sobreviven los grupos densos: los vínculos genuinos al azar forman cadenas sueltas, las operaciones forman redes cerradas.

## Dónde guardamos los datos y por qué

**DuckDB en proceso sobre archivos Parquet.**

- **Parquet** guarda cada columna junta, comprimida y con tipos: los 200.000 comentarios ocupan una fracción del CSV, y una consulta que lee dos columnas de un día no toca el resto. Es el formato estándar de los *data lakes* y lo leen DuckDB, Spark, ClickHouse, BigQuery y pandas.
- **DuckDB** es una base analítica que corre dentro del proceso de Python, sin servidor: lee CSV y Parquet directo con SQL y usa todos los núcleos. La usamos para la ingesta y la limpieza y para las consultas pesadas.
- **Por qué no otra cosa:** un servidor (PostgreSQL, Elasticsearch) agrega instalación y operación sin ganar nada a esta escala, y el jurado tendría que levantarlo para reproducir la corrida. Con DuckDB, reproducir es `make instalar && make final`.

**Cómo escala a millones de comentarios por día.** El diseño ya es incremental:

- Los comentarios llegan como Parquet particionado por día en almacenamiento de objetos; la misma consulta DuckDB corre ahí, o pasa a MotherDuck o ClickHouse sin cambiar el formato.
- Solo se embeben textos nuevos: la caché por texto absorbe repeticiones y campañas de copiar y pegar. A unos 300 textos por segundo por notebook, y paralelizable por partición.
- Los comentarios nuevos se asignan a la narrativa del centroide más cercano. Los que no encajan se acumulan y se re-agrupan cada pocas horas: así aparecen las narrativas emergentes, con su fecha de origen.
- La coordinación usa ventanas deslizantes sobre el grafo de co-actividad, que solo crece con los pares de cuentas que coinciden.
- La IA se usa sobre muestras por narrativa, no por comentario: su costo depende de cuántas narrativas hay, no de cuántos comentarios.

## Uso de IA y costo

La IA se usa donde suma y no como único motor:

- una llamada para nombrar y revisar narrativas;
- dos rondas de etiquetado de ~30 comentarios por narrativa para la postura.

Todo lo demás corre local y gratis. `costo.json` registra llamadas, tokens y dólares de cada corrida. En el set de práctica:

| Modelo | Llamadas | Tokens | Costo de la corrida | Cada 1.000 comentarios | Postura |
|---|---|---|---|---|---|
| Claude Haiku 5.5 (por defecto) | 27 | ~62.000 | US$ 0,012 | US$ 0,00006 | 0,96 |
| Claude Opus 5.5 | 27 | ~61.000 | US$ 0,47 | US$ 0,0024 | 0,96 |

Haiku rinde lo mismo por la cuadragésima parte. A modo de referencia, mandar cada comentario a un modelo costaría decenas de dólares y horas de llamadas; acá la IA mira unos 1.000 comentarios en total y suma unos 25 segundos a la corrida.

## Resultados en el set de práctica

Medidos con el script oficial (`desafio-qualydata/evaluacion/puntuar.py`):

| Corrida | Narrativas (30) | Coordinación (25) | Total (55) |
|---|---|---|---|
| Línea base del desafío | 12,7 | 4,1 | 16,8 |
| Radar, sin IA | 27,4 | 24,0 | 51,4 |
| **Radar con Claude Haiku 5.5** (por defecto) | 29,8 | 24,0 | **53,8** |
| Radar con Claude Opus 5.5 | 29,7 | 24,0 | 53,7 |
| Radar, postura con etiquetas perfectas (techo del método) | 29,9 | 24,0 | 53,9 |

Detalle:
- **Narrativas:** agrupamiento B-cubed 0,999, ruido 0,999, postura 0,96, origen de las emergentes 1,0 (las tres a menos de 3 horas), 13 narrativas encontradas de 13 reales.
- **Coordinación:** precisión 1,000, cobertura 0,939, agrupamiento por operación 0,942. Las tres campañas encontradas, ninguna cuenta genuina acusada.

**Generalización** (sin ajustar nada por tema, `make rodajas`, con Haiku, dos corridas):

| Rodaja | Narrativas reales / encontradas | Agrupamiento | Postura | Coordinación | Total |
|---|---|---|---|---|---|
| Litio (2 campañas) | 5 / 5 | 0,993 | 0,93–0,94 | F1 0,983 | 54,0–54,1 |
| Colectivos (sin campañas) | 4 / 4 | 0,996 | 0,97 | 0 cuentas marcadas (correcto) | 29,7 de ~30 posibles |
| Hospital (1 campaña) | 4 / 4 | 1,000 | 0,97 | F1 1,000 | 54,8 |

En litio y hospital el agrupamiento deja un fragmento chico de una narrativa que cambia de redacción (49 y 152 comentarios); la revisión de Claude lo reconoce y lo fusiona.

**Robustez** (`make sucio`, datos corrompidos a propósito): 53,4 contra 53,7 con los datos limpios.

**Postura frente a errores de la IA:** con etiquetas perfectas la postura llega a 0,988; con 10% de etiquetas equivocadas, a 0,956; con 20%, a 0,913. Por eso un modelo chico alcanza.

## Validación

Los scripts de `eval/` documentan cómo se tomó cada decisión:

| Script | Qué mide |
|---|---|
| `slices.py` | Parte la práctica por tema para probar en temas no vistos |
| `tune_narratives.py` | Compara configuraciones de agrupamiento contra la verdad |
| `probe_nli_stance.py`, `probe_embed_stance.py` | Postura: modelo NLI sin entrenar (0,71) contra clasificador por narrativa (0,99 con 60 etiquetas) |
| `probe_label_selection.py` | Qué comentarios conviene etiquetar (aleatorio, diverso, aprendizaje activo, adaptativo) |
| `probe_stance_pipeline.py` | Postura dentro del pipeline con etiquetas perfectas, con ruido y sin IA |
| `explore_coordination.py` | Qué señales separan campañas de grupos genuinos |
| `make_dirty.py` | Genera una copia corrupta de los datos |
| `diagnose.py` | Explica un puntaje: cómo se reparte cada narrativa real |

## Qué existía antes del evento

Todo el código de este repositorio se escribió durante el hackathon. Usamos estas piezas públicas:

- **Librerías:** DuckDB, PyArrow, pandas, NumPy, scikit-learn (k-means, regresión logística), sentence-transformers y PyTorch (embeddings en CPU), python-igraph y leidenalg (grafos y detección de comunidades), SciPy y el SDK de Anthropic. Versiones exactas en `uv.lock`.
- **Modelos:** `intfloat/multilingual-e5-small` (embeddings, abierto, local) y Claude (Anthropic) por API.
- **Material del desafío:** datos, línea base y script de puntaje de [qualydata-org/desafio-qualydata](https://github.com/qualydata-org/desafio-qualydata). No se usaron otros datos.

## Licencia

[AGPL-3.0 o posterior](LICENSE). Cualquiera puede usar, estudiar, modificar y redistribuir este código. Quien lo modifique y lo ofrezca como servicio en red tiene que publicar sus cambios bajo la misma licencia.

## Límites

- Parte de la fuerza viene de señales estructurales de esta conversación (las respuestas discuten dentro de la narrativa de su padre, el ruido casi nunca recibe respuestas). El pipeline las usa como señales y no como reglas fijas, y mide cuánto depende de ellas.
- Sin IA, la postura es débil (0,57): separa dos lados por la dirección en que las respuestas difieren de su padre, pero a veces confunde cuál es cuál.
- La coordinación encuentra grupos que actúan juntos. Una operación de una sola cuenta, o una que imita perfectamente el ritmo genuino, no deja ese rastro.
