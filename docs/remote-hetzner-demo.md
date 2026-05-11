# Remote Hetzner Demo

This path runs the Blinkin VLM UI/backend on a normal CPU server and calls hosted
model APIs for the expensive work.

```text
Browser
  -> Hetzner FastAPI app
      -> Google Gemini for prompt-to-target coordinates
      -> Replicate for Depth Anything metric depth
  -> Three.js point cloud in the browser
```

## 1. Add Secrets

Copy the example file:

```bash
cp .env.example .env
```

Fill in these values on the server:

```bash
GEMINI_API_KEY=...
REPLICATE_API_TOKEN=...
```

Do not commit `.env`. It is ignored by `.gitignore`.

## 2. Run With Docker Compose

```bash
docker compose up -d --build
```

Open:

```text
http://SERVER_IP:8080
```

Health check:

```bash
curl http://127.0.0.1:8080/healthz
```

## 3. Put HTTPS In Front

Use Caddy or Nginx as the public HTTPS reverse proxy. The app listens on local
port `8080` by default.

Example Caddy shape:

```text
your-domain.example {
  reverse_proxy 127.0.0.1:8080
}
```

## Notes

- The server does not need a GPU in remote mode.
- Replicate and Gemini calls cost money per use. Keep the demo behind a private
  URL or auth gate before sharing widely.
- The default Replicate model is `david20321/depth-anything-v3-metric-large`
  because it returns a metric 16-bit depth PNG plus scale metadata.
- The default pinned Replicate version is
  `e3523ab17a5e6f0e279933a6afdde67efe130bb9e7753cafc52a4b082257f46b`.
- If you swap Replicate models, keep the output compatible with either:
  `depth_png_base64`, `depth_png` with metric metadata, an NPZ depth array, or a
  depth image fallback.
