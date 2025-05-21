# Réplica del paper "Weakly Supervised Video Anomaly Detection Based on Hyperbolic Space"

Este repositorio es un tipo de réplica del código del artículo:
**"Detección de Anomalías de Vídeo con Supervisión Débil Basada en el Espacio Hiperbólico"**
de Meilin Qi y Yuanyuan Wu (2024).

Artículo original: [DOI](https://doi.org/10.1038/s41598-024-77505-4) | [Nature Scientific Reports](https://www.nature.com/articles/s41598-024-77505-4).

## Licencia
Esta obra está licenciada bajo la licencia [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), al igual que el artículo original.
**Nota**: Este repositorio es solo para fines educativos y no comerciales.

# Detección de Violencia Física en Videos  
**Usando Redes Neuronales en Espacio Hiperbólico**

## Desarrolladores de codigo
- **Anayeli Castro Giménez**
- **José Ángel Rodríguez Estrada**
- **Iván Zainos Ascencio**

*Instituto Politécnico Nacional (IPN)*  
 `acastrog2200@alumno.ipn.mx`, `jrodrigueze2200@alumno.ipn.mx`, `izainosa2200@alumno.ipn.mx`  

---

## Conjunto de Datos
Se utilizó la base de datos **UCF-Crime**, un conjunto de videos reales que contienen diversas actividades anómalas. Para esta implementación, se trabajó específicamente con la clase de *violencia física*, usando características extraídas previamente con el modelo **I3D (Inflated 3D ConvNet)**.

---

## Metodología
Se propone un sistema de detección de anomalías basado en grafos y aprendizaje débilmente supervisado:

- **Grafo de Similitud**: Modela la similitud coseno entre frames.
- **Grafo Temporal**: Representa relaciones basadas en la distancia temporal entre frames.

Ambos grafos se procesan mediante una red neuronal en espacio **hiperbólico** usando capas **GCN (Graph Convolutional Network)**.

---

## Modelo
El modelo se compone de:

- Dos capas GCN hiperbólicas (`geoopt.PoincareBall`) que operan sobre los grafos construidos.
- Una capa completamente conectada que predice un score de anomalía por frame.
- Uso del mapeo logarítmico y exponencial para transformar los datos al espacio hiperbólico.
