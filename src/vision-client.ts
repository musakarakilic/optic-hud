/**
 * WebSocket Client for Local NVIDIA GPU Vision Server (YOLOE-11s-seg + SAM 2.1 Tiny)
 * - Streams compressed camera frames with strict backpressure (1 in-flight frame at a time -> zero lag!)
 * - Receives exact 0..1000 polygon silhouettes + bounding boxes in real time on PC & Android Chrome.
 */

export interface GpuVisionResult {
  found: boolean;
  mode: 'text' | 'sam_memory' | 'idle';
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  polygon?: Array<[number, number]>;
  score?: number;
  ms?: number;
  model?: string;
}

export class GpuVisionClient {
  private ws: WebSocket | null = null;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private active = false;
  private inFlight = false;
  private rafId: number | null = null;
  private lastSentMs = 0;

  public connected = false;
  public gpuName = '';

  private onResult: (res: GpuVisionResult) => void;
  private onStatus?: (text: string) => void;

  constructor(onResult: (res: GpuVisionResult) => void, onStatus?: (text: string) => void) {
    this.onResult = onResult;
    this.onStatus = onStatus;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
  }

  get isTracking(): boolean {
    return this.active;
  }

  attachVideo(video: HTMLVideoElement): void {
    this.video = video;
  }

  async checkHealth(): Promise<{ ok: boolean; gpu?: string; device?: string }> {
    try {
      const res = await fetch('/vision-api/health');
      if (!res.ok) return { ok: false };
      const data = (await res.json()) as { ok: boolean; gpu?: string; device?: string };
      this.gpuName = data.gpu || '';
      return data;
    } catch {
      return { ok: false };
    }
  }

  private ensureSocket(): Promise<boolean> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return Promise.resolve(true);
    }
    return new Promise((resolve) => {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${window.location.host}/vision-ws`;
      const ws = new WebSocket(url);
      let settled = false;

      const finish = (ok: boolean) => {
        if (!settled) {
          settled = true;
          resolve(ok);
        }
      };

      ws.onopen = () => {
        this.ws = ws;
        this.connected = true;
        this.inFlight = false;
        finish(true);
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(String(ev.data)) as Record<string, unknown>;
          if (msg.type === 'result') {
            this.inFlight = false;
            this.onResult({
              found: Boolean(msg.found),
              mode: (msg.mode as GpuVisionResult['mode']) || 'idle',
              x: typeof msg.x === 'number' ? msg.x : undefined,
              y: typeof msg.y === 'number' ? msg.y : undefined,
              width: typeof msg.width === 'number' ? msg.width : undefined,
              height: typeof msg.height === 'number' ? msg.height : undefined,
              polygon: Array.isArray(msg.polygon) ? (msg.polygon as Array<[number, number]>) : [],
              score: typeof msg.score === 'number' ? msg.score : undefined,
              ms: typeof msg.ms === 'number' ? msg.ms : undefined,
              model: typeof msg.model === 'string' ? msg.model : undefined,
            });
          } else if (msg.type === 'error') {
            this.inFlight = false;
            this.onStatus?.(`GPU Vision error: ${String(msg.message ?? '')}`);
          }
        } catch {
          this.inFlight = false;
        }
      };

      ws.onerror = () => {
        this.connected = false;
        this.inFlight = false;
        finish(false);
      };

      ws.onclose = () => {
        this.connected = false;
        this.inFlight = false;
        if (this.ws === ws) this.ws = null;
        finish(false);
      };

      setTimeout(() => finish(false), 3000);
    });
  }

  async trackText(englishPrompt: string): Promise<boolean> {
    const ok = await this.ensureSocket();
    if (!ok || !this.ws) return false;
    this.active = true;
    this.inFlight = false;
    this.ws.send(JSON.stringify({ cmd: 'track_text', prompt: englishPrompt }));
    this.startLoop();
    return true;
  }

  async trackPoint(nx: number, ny: number): Promise<boolean> {
    const ok = await this.ensureSocket();
    if (!ok || !this.ws) return false;
    this.active = true;
    this.inFlight = false;
    this.ws.send(JSON.stringify({ cmd: 'track_point', x: nx, y: ny }));
    this.startLoop();
    return true;
  }

  lockCurrentToSam(): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.active = true;
    this.inFlight = false;
    this.ws.send(JSON.stringify({ cmd: 'lock_current' }));
    this.startLoop();
    return true;
  }

  stop(): void {
    this.active = false;
    this.inFlight = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ cmd: 'clear' }));
    }
  }

  private startLoop(): void {
    if (this.rafId !== null) return;
    this.rafId = requestAnimationFrame(this.tick);
  }

  private tick = (now: number): void => {
    if (!this.active) {
      this.rafId = null;
      return;
    }
    this.rafId = requestAnimationFrame(this.tick);

    // Backpressure: wait for previous GPU frame result + cap at ~20 FPS for cool GPU temps
    if (this.inFlight || now - this.lastSentMs < 48) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (!this.video || this.video.readyState < 2 || !this.video.videoWidth) return;

    const maxDim = 512;
    const scale = Math.min(1, maxDim / Math.max(this.video.videoWidth, this.video.videoHeight));
    this.canvas.width = Math.round(this.video.videoWidth * scale);
    this.canvas.height = Math.round(this.video.videoHeight * scale);
    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    const jpeg = this.canvas.toDataURL('image/jpeg', 0.72).split(',')[1];
    if (!jpeg) return;

    this.inFlight = true;
    this.lastSentMs = now;
    this.ws.send(JSON.stringify({ cmd: 'frame', jpeg }));
  };
}
