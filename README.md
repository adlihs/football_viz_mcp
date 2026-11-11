# football_viz_mcp

Servidor **MCP (Model Context Protocol)** que expone los gráficos de la librería
[mplsoccer](https://mplsoccer.readthedocs.io) como herramientas. Cualquier LLM compatible con MCP
(Claude, ChatGPT, Cursor, VS Code, modelos locales…) puede recibir los datos del usuario y responder
con un gráfico de fútbol en PNG.

- **Remoto (HTTP)**: transporte *Streamable HTTP* en `/mcp`. También funciona local por `stdio`.
- **Datos del usuario**: CSV, Excel, JSON (incluye eventos StatsBomb anidados), URL pública o Google Sheets.
- **Cubre todo mplsoccer**: todos los tipos de cancha, todos los métodos de dibujo y Radar, Pizza y Bumpy.

---

## 1. Herramientas (22)

| Grupo | Herramienta | Qué hace |
|---|---|---|
| Datos | `load_dataset` | Carga los datos del usuario (csv_text, json_data, url, file_base64) y devuelve un `dataset_id` |
| | `describe_dataset` | Columnas, tipos, rangos, valores de categorías y **pitch_type sugerido** |
| | `transform_dataset` | Filtrar, renombrar, convertir coordenadas entre proveedores (`Standardizer`), ángulo/distancia, top N |
| | `list_datasets`, `delete_dataset` | Gestión |
| Referencia | `get_reference`, `list_formations` | Documentación de todas las opciones para el LLM |
| Cancha | `draw_pitch` | Cancha vacía (todos los estilos) |
| | `shot_map` | Mapa de remates: tamaño por xG, color por resultado, goles como balón, ángulo de tiro, KDE |
| | `pass_map` | Pases/conducciones con flechas, líneas o *comets*, color por resultado/jugador/valor |
| | `heatmap` | Mapas de calor: grilla, **Juego de Posición**, zonas propias, KDE, hexbin; % o cualquier estadística |
| | `flow_map` | Dirección media de los pases por zona sobre un mapa de calor |
| | `pass_network` | Red de pases (posiciones medias + conexiones) y las combinaciones más frecuentes |
| | `pass_sonar` | Sonares de pase por zona (frecuencia y distancia por dirección) |
| | `freeze_frame` | Tracking / freeze-frame: Voronoi, Delaunay, convex hull, balón y ángulo de tiro |
| | `formation_chart` | Alineación de un equipo o dos enfrentados (68 formaciones) |
| | `joint_plot` | Cancha con distribuciones marginales |
| | `pitch_chart` | **Composición libre**: apila cualquiera de las 22 capas y facetas (*small multiples*) |
| | `zone_stats` | Solo números (sin imagen) por zona, para responder en texto |
| Otros | `radar_chart` | Radar estándar, sólido o *turbine*; comparación de 2 jugadores; rangos automáticos |
| | `pizza_chart` | Pizza de percentiles, por categorías o comparando jugadores |
| | `bumpy_chart` | Evolución de posiciones en la tabla por jornada |

**Capas de `pitch_chart`**: `scatter, arrows, lines, comet, plot, kde, hexbin, heatmap, heatmap_positional,
heatmap_zones, zones, flow, sonar, pass_network, voronoi, delaunay, convexhull, goal_angle, text, annotate,
polygon, formation`. Cada capa puede tener su propio dataset y filtros. Con `figure.facet_by` se dibuja
una cancha por jugador, tiempo o partido.

**Tipos de cancha**: `statsbomb, opta, wyscout, uefa, metricasports, custom, tracab, skillcorner,
secondspectrum, impect`. **Temas**: `light, dark, grass, night`.

Cada herramienta devuelve la **imagen PNG** y un **resumen en texto** con los números graficados
(tiros, goles, xG, conexiones más frecuentes…) para que el LLM pueda explicar el gráfico. Si los puntos caen
fuera de la cancha, el resumen avisa que el `pitch_type` probablemente es incorrecto.

---

## 2. Instalación y ejecución local

```bash
pip install -r requirements.txt
python -m football_viz_mcp            # HTTP en http://localhost:8000/mcp
python -m football_viz_mcp --stdio    # modo stdio (Claude Desktop, Cursor...)
```

Prueba completa (llama a todas las herramientas por HTTP y guarda las imágenes):

```bash
python examples/make_sample_data.py
python tests/smoke_test.py "http://localhost:8000/mcp" salida/
```

## 3. Despliegue remoto

Con Docker (Render, Railway, Fly.io, Google Cloud Run, un VPS…):

```bash
docker build -t football-viz-mcp .
docker run -p 8000:8000 -e MCP_API_KEY=tu-clave -e PUBLIC_BASE_URL=https://tu-dominio football-viz-mcp
```

En **Render** basta con conectar el repositorio: el archivo `render.yaml` ya define el servicio y genera la
clave. Pon el servidor detrás de HTTPS (todas estas plataformas lo hacen automáticamente).

### Cloudflare (Workers + Containers + R2)

La carpeta `cloudflare/` tiene todo lo necesario. Un **Worker** recibe las peticiones, revisa la clave
(sin despertar al contenedor) y las pasa a un **Container** que ejecuta la imagen de este `Dockerfile`.
Los datasets y gráficos se guardan en **R2**, así que no se pierden cuando el contenedor se duerme por
inactividad: solo hay un arranque en frío de unos segundos en la siguiente petición.

**Requisitos**: plan *Workers Paid* (5 USD/mes, incluye 25 GiB-hora de memoria y 375 minutos de vCPU al mes),
**Docker Desktop** abierto en tu computadora (wrangler construye la imagen ahí) y Node.js 20 o superior.

```bash
cd cloudflare
npm install
npx wrangler login                                   # abre el navegador para autorizar tu cuenta

# 1) Bucket R2 para los datasets y gráficos
npx wrangler r2 bucket create football-viz-mcp
```

2) **Credenciales de R2**: en el panel de Cloudflare → *R2* → *Manage API tokens* → *Create API token*,
permiso **Object Read & Write**, limitado al bucket `football-viz-mcp`. Copia el *Access Key ID* y el
*Secret Access Key* (el secreto solo se muestra una vez). Tu *Account ID* aparece en la misma página de R2.

3) Pon tu Account ID en `cloudflare/wrangler.jsonc` → `vars.R2_ACCOUNT_ID`.

4) Guarda los secretos (te los pide uno por uno; no van en ningún archivo):

```bash
npx wrangler secret put MCP_API_KEY            # inventa una clave larga
npx wrangler secret put R2_ACCESS_KEY_ID
npx wrangler secret put R2_SECRET_ACCESS_KEY
```

5) Despliega:

```bash
npx wrangler deploy
```

Al terminar muestra la URL, por ejemplo `https://football-viz-mcp.tu-subdominio.workers.dev`.
Cópiala en `vars.PUBLIC_BASE_URL` del `wrangler.jsonc` y ejecuta `npx wrangler deploy` otra vez para que
cada gráfico incluya su enlace de descarga. **Espera unos minutos** después del primer despliegue: el
contenedor tarda en aprovisionarse y mientras tanto las peticiones pueden fallar.

6) Prueba: abre `https://…workers.dev/health` (debe responder `{"status":"ok"}`; la primera vez tarda unos
segundos porque despierta el contenedor). La URL para conectar el cliente MCP es
`https://…workers.dev/mcp?key=TU_CLAVE` y la página de carga `https://…workers.dev/upload?key=TU_CLAVE`.

**Ajustes en `wrangler.jsonc`**

| Ajuste | Por defecto | Para qué |
|---|---|---|
| `instance_type` | `standard-1` (½ vCPU, 4 GiB) | `basic` (¼ vCPU, 1 GiB) es más barato y alcanza para datasets pequeños |
| `vars.SLEEP_AFTER` | `30m` | Minutos de inactividad antes de que el contenedor se duerma (y deje de cobrar) |
| `vars.INSTANCES` | `1` | Más de 1 reparte la carga entre varios contenedores (funciona porque los datos están en R2) |
| `vars.DATASET_TTL_HOURS` | `168` | Los datasets se borran a los 7 días |

**Comandos útiles**: `npx wrangler tail` (ver logs en vivo), `npx wrangler containers list`
(estado de los contenedores). Para que los gráficos viejos se borren solos, en el panel de R2 →
bucket → *Settings* → *Object lifecycle rules*, agrega una regla para el prefijo `football-viz-mcp/charts/`
(por ejemplo, borrar a los 30 días).

### Variables de entorno

| Variable | Por defecto | Uso |
|---|---|---|
| `MCP_API_KEY` | (vacío = sin clave) | Exige `Authorization: Bearer <clave>`, `X-API-Key` o `?key=<clave>` |
| `PUBLIC_BASE_URL` | — | Agrega un enlace de descarga `…/charts/<id>.png` a cada gráfico (los enlaces son públicos pero imposibles de adivinar) |
| `PORT` / `HOST` | 8000 / 0.0.0.0 | |
| `R2_BUCKET_NAME`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | — | Guarda datasets y gráficos en Cloudflare R2 (sobreviven reinicios) |
| `S3_ENDPOINT_URL`, `S3_PREFIX` | — / `football-viz-mcp` | Para usar otro almacenamiento compatible con S3 en vez de R2 |
| `FOOTBALL_VIZ_MCP_DATA_DIR` | — | Alternativa sin R2: carpeta en disco para los datasets |
| `FOOTBALL_VIZ_MCP_DATASET_TTL_HOURS` | 72 | Tiempo de vida de un dataset |
| `FOOTBALL_VIZ_MCP_MAX_ROWS` | 500000 | Límite de filas por dataset |
| `FOOTBALL_VIZ_MCP_MAX_BODY_MB` | 25 | Tamaño máximo de una petición MCP |
| `FOOTBALL_VIZ_MCP_ALLOW_URLS` | true | Permitir cargar datos desde URLs (se bloquean IPs privadas) |

## 4. Conectar un cliente

URL del servidor: `https://tu-dominio/mcp` (con clave: `https://tu-dominio/mcp?key=tu-clave`).

- **Claude (web / escritorio)**: Configuración → Conectores → *Agregar conector personalizado* → pega la URL
  (usa la variante `?key=` porque ese formulario no permite cabeceras).
- **Claude Code**: `claude mcp add --transport http football_viz_mcp https://tu-dominio/mcp --header "Authorization: Bearer tu-clave"`
- **Cursor / VS Code / otros**: en la configuración MCP del cliente
  ```json
  { "mcpServers": { "football_viz_mcp": { "url": "https://tu-dominio/mcp",
      "headers": { "Authorization": "Bearer tu-clave" } } } }
  ```
- **Claude Desktop local (stdio)**:
  ```json
  { "mcpServers": { "football_viz_mcp": { "command": "python", "args": ["-m", "football_viz_mcp", "--stdio"],
      "cwd": "/ruta/a/football-viz-mcp" } } }
  ```

## 5. Cómo entrega el usuario sus datos

1. **Pegando o adjuntando** el CSV/JSON en el chat: el LLM llama a `load_dataset`.
2. **Página de carga** `https://tu-dominio/upload?key=tu-clave`: el usuario sube un archivo grande (Excel, CSV,
   JSON de StatsBomb) y le dice al asistente el `dataset_id`. Así el archivo no pasa por el chat ni gasta tokens.
   También por API: `curl -X POST "https://tu-dominio/upload?key=…&name=partido&filename=p.csv" --data-binary @p.csv`
3. **Un enlace** a un CSV público o a una hoja de Google Sheets compartida ("cualquiera con el enlace").

El servidor detecta separadores `;` y decimales con coma (habitual en Excel en español), aplana JSON anidado
(`pass.end_location` → `pass.end_location_x/_y`) y detecta solas las columnas de coordenadas más comunes
(`x, y, end_x, end_y, location_x…`).

## 6. Ejemplos de lo que puede pedir el usuario

- "Con este CSV, haz el mapa de remates de Saprissa con el tamaño según el xG."
- "Red de pases de Alajuelense, solo pases completos, mínimo 4 pases por conexión."
- "Mapa de calor por zonas de Juego de Posición con porcentajes, una cancha por tiempo."
- "Compara en un radar a Campbell y Ugalde usando solo delanteros con más de 900 minutos."
- "¿Qué porcentaje de nuestras acciones fueron en el último tercio?" (→ `zone_stats`)

Ejemplo de llamada a `pitch_chart` (la llamada la escribe el LLM, no el usuario):

```json
{
  "dataset_id": "ds_ab12cd34ef",
  "pitch": {"pitch_type": "statsbomb", "positional": true},
  "figure": {"title": "Pases al último tercio", "theme": "dark", "facet_by": "half"},
  "layers": [
    {"type": "kde", "filters": [{"column": "type", "value": "Pass"}], "alpha": 0.5},
    {"type": "comet", "filters": [{"column": "type", "value": "Pass"},
                                  {"column": "end_x", "op": ">", "value": 80}], "color_by": "outcome"}
  ]
}
```

## 7. Estructura

```
football_viz_mcp/
  server.py        herramientas MCP
  pitch_charts.py  motor de capas sobre la cancha (todas las funciones de Pitch/VerticalPitch)
  other_charts.py  Radar, PyPizza, Bumpy, jointgrid
  data.py          carga, almacenamiento, filtros y detección de columnas
  render.py        temas, construcción de la cancha, exportar PNG
  reference.py     documentación que se entrega al LLM
  app.py           app HTTP: /mcp, /upload, /charts/<id>.png, /health, clave de acceso
cloudflare/        Worker (src/index.ts) + wrangler.jsonc para desplegar en Cloudflare Containers
examples/          generador de datos de ejemplo (coordenadas StatsBomb)
tests/smoke_test.py prueba de extremo a extremo
```

Requiere Python ≥ 3.11, `mcp` 2.x (SDK oficial de Python) y `mplsoccer` ≥ 1.8.1.
