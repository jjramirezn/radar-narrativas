DESAFIO := desafio-qualydata
PRACTICA := $(DESAFIO)/datos/practica
PUNTUAR := uv run python $(DESAFIO)/evaluacion/puntuar.py

.PHONY: instalar datos modelos final practica sucio rodajas

instalar:  ## Crea el entorno con las versiones exactas de uv.lock (Python 3.12, sin GPU)
	uv sync --frozen

datos:  ## Baja el material del desafío (set de práctica y script de puntaje)
	test -d $(DESAFIO) || git clone --depth 1 https://github.com/qualydata-org/desafio-qualydata.git $(DESAFIO)

modelos:  ## Baja el modelo de embeddings una vez, para que la corrida no dependa de internet
	uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', allow_patterns=['*.json', '*.safetensors', '*.model', '1_Pooling/*'])"

final:  ## Corre sobre el set final (datos/final) y deja el resultado en entrega/
	uv run radar --datos datos/final --entrega entrega

practica:  ## Corre sobre el set de práctica y lo puntúa con el script oficial
	uv run radar --datos $(PRACTICA) --entrega salida/practica
	$(PUNTUAR) --verdad $(PRACTICA)/verdad --entrega salida/practica

sucio:  ## Robustez: corrompe una copia de la práctica a propósito, corre y puntúa
	uv run python eval/make_dirty.py --datos $(PRACTICA) --salida salida/sucio
	uv run radar --datos salida/sucio --entrega salida/sucio-entrega
	$(PUNTUAR) --verdad salida/sucio/verdad --entrega salida/sucio-entrega

rodajas:  ## Generalización: corre y puntúa cada tema de la práctica por separado
	uv run python eval/slices.py --datos $(PRACTICA) --salida salida/slices
	for tema in litio colectivo hospital; do \
		uv run radar --datos salida/slices/$$tema --entrega salida/slices-entrega/$$tema && \
		$(PUNTUAR) --verdad salida/slices/$$tema/verdad --entrega salida/slices-entrega/$$tema; \
	done
