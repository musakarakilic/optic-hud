import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI } from '@google/genai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = Number(process.env.PORT || 3001);
const DEBUG_LOG_PATH = path.resolve(__dirname, '..', 'debug.log');

app.use(express.json({ limit: '2mb' }));

app.post('/api/log', (req, res) => {
  const { event, detail } = (req.body ?? {}) as { event?: string; detail?: unknown };
  const line = `[${new Date().toISOString()}] ${event ?? 'event'} ${detail !== undefined ? JSON.stringify(detail) : ''}`;
  console.log('[CLIENT-DIAG]', line);
  try {
    fs.appendFileSync(DEBUG_LOG_PATH, line + '\n', 'utf8');
  } catch {
    // ignore
  }
  res.json({ ok: true });
});

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    hasApiKey: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'your_gemini_api_key_here'),
  });
});

// Single cloud model: Gemini 3.8 Live ephemeral token
app.post('/api/token', async (_req, res) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'your_gemini_api_key_here') {
    res.status(500).json({
      error: 'GEMINI_API_KEY is not configured. Add it to ai-glasses-mvp/.env.',
    });
    return;
  }

  try {
    const ai = new GoogleGenAI({ apiKey, httpOptions: { apiVersion: 'v1beta' } });
    const expireTime = new Date(Date.now() + 30 * 60 * 1000).toISOString();
    const newSessionExpireTime = new Date(Date.now() + 5 * 60 * 1000).toISOString();

    const token = await ai.authTokens.create({
      config: {
        uses: 1,
        expireTime,
        newSessionExpireTime,
      },
    });

    res.json({
      token: token.name,
      model: 'gemini-3.8-live',
    });
  } catch (err) {
    console.error('Ephemeral token creation failed:', err);
    res.status(500).json({
      error: err instanceof Error ? err.message : 'Failed to create ephemeral token.',
    });
  }
});

const distDir = path.resolve(__dirname, '..', 'dist');
app.use(express.static(distDir));
app.get('*', (_req, res, next) => {
  if (_req.path.startsWith('/api')) return next();
  res.sendFile(path.join(distDir, 'index.html'), (err) => {
    if (err) res.status(404).send('Frontend build not found. Use npm run dev for development.');
  });
});

app.listen(PORT, () => {
  console.log(`[OPTIC HUD Server] listening on http://localhost:${PORT}`);
});
