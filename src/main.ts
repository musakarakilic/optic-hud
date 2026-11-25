import {
  Behavior,
  GoogleGenAI,
  Modality,
  Type,
  type FunctionDeclaration,
  type LiveServerMessage,
} from '@google/genai';
import { AudioStreamer } from './audio';
import { ArOverlayController, type ArAction, type ArColor } from './ar-overlay';
import { HudController } from './hud';
import { GpuVisionClient } from './vision-client';

const videoEl = document.getElementById('camera-video') as HTMLVideoElement;
const canvasEl = document.getElementById('capture-canvas') as HTMLCanvasElement;
const arOverlayEl = document.getElementById('ar-overlay') as unknown as SVGSVGElement;
const arLabelsEl = document.getElementById('ar-labels') as HTMLDivElement;
const btnClearAr = document.getElementById('btn-clear-ar') as HTMLButtonElement | null;
const btnConnect = document.getElementById('btn-connect') as HTMLButtonElement;
const btnMute = document.getElementById('btn-mute') as HTMLButtonElement;
const btnSwitchCam = document.getElementById('btn-switch-cam') as HTMLButtonElement;
const btnTestAudio = document.getElementById('btn-test-audio') as HTMLButtonElement | null;
const micSelect = document.getElementById('mic-select') as HTMLSelectElement | null;
const micLevelBar = document.getElementById('mic-level-fill') as HTMLDivElement | null;
const txCounterEl = document.getElementById('tx-counter') as HTMLSpanElement | null;
const textForm = document.getElementById('text-form') as HTMLFormElement;
const textInput = document.getElementById('text-input') as HTMLInputElement;
const btnSend = document.getElementById('btn-send') as HTMLButtonElement;
const yoloeForm = document.getElementById('yoloe-form') as HTMLFormElement | null;
const yoloeInput = document.getElementById('yoloe-input') as HTMLInputElement | null;
const btnYoloeLock = document.getElementById('btn-yoloe-lock') as HTMLButtonElement | null;
const gpuBadgeEl = document.getElementById('gpu-badge') as HTMLSpanElement | null;

const hud = new HudController();
const audio = new AudioStreamer();

let gpuMissStreak = 0;
let lastGpuLogMs = 0;

function updateGpuBadge(text: string, state: 'ready' | 'busy' | 'off' = 'ready'): void {
  if (!gpuBadgeEl) return;
  gpuBadgeEl.textContent = text;
  gpuBadgeEl.className = `gpu-readout${state === 'busy' ? ' gpu-busy' : state === 'off' ? ' gpu-off' : ''}`;
}

function setConnectButton(active: boolean): void {
  btnConnect.classList.toggle('active', active);
  const label = btnConnect.querySelector('.btn-connect-label');
  if (label) label.textContent = active ? 'End Live' : 'Start Live';
}

function setMuteButton(muted: boolean): void {
  btnMute.classList.toggle('is-muted', muted);
  const label = btnMute.querySelector('.btn-label');
  if (label) label.textContent = muted ? 'Muted' : 'Mic On';
}

const gpuVision = new GpuVisionClient(
  (res) => {
    if (res.found && res.x !== undefined && res.y !== undefined) {
      gpuMissStreak = 0;
      arOverlay.updatePrimaryTarget({
        x: res.x,
        y: res.y,
        boxWidth: res.width ?? 100,
        boxHeight: res.height ?? 100,
        polygon: res.polygon ?? [],
        trackingState: 'locked',
        confidence: res.score,
      });
      if (btnYoloeLock) {
        btnYoloeLock.classList.toggle('hidden', res.mode !== 'text');
      }
      const tag = res.mode === 'sam_memory' ? 'SAM2-MEM' : 'YOLOE-11M';
      updateGpuBadge(`${tag} ${res.ms ?? 0}ms`, 'ready');
      const now = performance.now();
      if (now - lastGpuLogMs > 4000) {
        lastGpuLogMs = now;
        sendDiag('gpu_track', {
          model: res.model,
          mode: res.mode,
          ms: res.ms,
          score: res.score,
          polyPts: res.polygon?.length ?? 0,
        });
      }
    } else {
      gpuMissStreak++;
      if (res.ms !== undefined && res.model) {
        const tag = res.mode === 'sam_memory' ? 'SAM2-MEM' : 'YOLOE-11M';
        updateGpuBadge(`SCAN ${tag} ${res.ms}ms`, 'busy');
      }
      if (gpuMissStreak > 4) {
        arOverlay.updatePrimaryTarget({
          x: 500,
          y: 500,
          boxWidth: 100,
          boxHeight: 100,
          polygon: [],
          trackingState: 'relocalizing',
        });
      }
    }
  },
  (statusMsg) => {
    hud.appendTranscript('system', statusMsg);
  }
);

const arOverlay = new ArOverlayController(arOverlayEl, arLabelsEl, videoEl, btnClearAr, () => {
  gpuVision.stop();
  btnYoloeLock?.classList.add('hidden');
  updateGpuBadge(
    gpuVision.gpuName ? gpuVision.gpuName.replace('NVIDIA GeForce ', '').toUpperCase() : 'GPU READY',
    'ready'
  );
});

let cameraStream: MediaStream | null = null;
let facingMode: 'environment' | 'user' = 'environment';
let frameTimer: number | null = null;
let liveSession: Awaited<ReturnType<GoogleGenAI['live']['connect']>> | null = null;
let isSessionReady = false;
let manualDisconnect = false;
let txPackets = 0;
let rxAudioChunks = 0;
let peakRms = 0;

function sendDiag(event: string, detail?: Record<string, unknown>): void {
  void fetch('/api/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event, detail }),
  }).catch(() => {});
}

// Focused, calm field-technician instruction in English for Gemini 3.8 Live + Local GPU Vision
const SYSTEM_INSTRUCTION = `You are OPTIC HUD, a live field repair and assembly assistant running on an AR headset.
Always speak in concise, calm, practical English like an experienced technician.
1. When starting a repair or setup procedure, call "set_task_plan" to display required tools and numbered steps on the HUD.
2. When the user completes a step, call "update_step" to mark it done.
3. When pointing out or highlighting a physical object or part in the camera view, call "point_to_target":
   - Put a short, clear visual English object name in "target_en" for the local GPU YOLOE-11m-seg tracker (e.g., "black earbud case", "white spray bottle", "bottle cap", "screw", "screwdriver", "keyboard key", "USB cable").
   - Put a concise HUD callout label in "label".`;

const HUD_TOOLS: FunctionDeclaration[] = [
  {
    name: 'set_task_plan',
    description: 'Displays the procedure title, required tools, and ordered steps on the AR HUD.',
    behavior: Behavior.BLOCKING,
    parameters: {
      type: Type.OBJECT,
      properties: {
        title: { type: Type.STRING, description: 'Short procedure title' },
        tools: { type: Type.ARRAY, items: { type: Type.STRING }, description: 'Required tools or parts' },
        steps: { type: Type.ARRAY, items: { type: Type.STRING }, description: 'Ordered concise steps' },
      },
      required: ['title', 'tools', 'steps'],
    },
  },
  {
    name: 'update_step',
    description: 'Marks a procedure step as completed.',
    behavior: Behavior.BLOCKING,
    parameters: {
      type: Type.OBJECT,
      properties: {
        step_number: { type: Type.INTEGER, description: '1-based step number' },
        done: { type: Type.BOOLEAN, description: 'True when completed' },
      },
      required: ['step_number', 'done'],
    },
  },
  {
    name: 'point_to_target',
    description:
      'Highlights and tracks an object in real time using the local NVIDIA GPU (YOLOE-11m-seg) with zero cloud vision quota.',
    behavior: Behavior.BLOCKING,
    parameters: {
      type: Type.OBJECT,
      properties: {
        target_en: {
          type: Type.STRING,
          description: 'Short visual English object name (e.g., "white spray bottle", "black earbud case", "screw", "screwdriver")',
        },
        label: { type: Type.STRING, description: 'Concise HUD callout label (e.g., Spray Bottle)' },
        action: { type: Type.STRING, description: 'point | turn_left | turn_right | press | pull | caution' },
        color: { type: Type.STRING, description: 'orange | green | amber | red | white' },
      },
      required: ['target_en', 'label'],
    },
  },
];

async function startCamera(): Promise<void> {
  if (!navigator.mediaDevices?.getUserMedia) {
    hud.appendTranscript('system', 'Camera API unavailable. Open via http://localhost:3000.');
    return;
  }
  cameraStream?.getTracks().forEach((t) => t.stop());
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
  videoEl.srcObject = cameraStream;
  gpuVision.attachVideo(videoEl);
}

function captureJpegBase64(maxDim = 768, quality = 0.72): string | null {
  if (!videoEl.videoWidth || !videoEl.videoHeight) return null;
  const scale = Math.min(1, maxDim / Math.max(videoEl.videoWidth, videoEl.videoHeight));
  canvasEl.width = Math.round(videoEl.videoWidth * scale);
  canvasEl.height = Math.round(videoEl.videoHeight * scale);
  const ctx = canvasEl.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
  const dataUrl = canvasEl.toDataURL('image/jpeg', quality);
  return dataUrl.split(',')[1] ?? null;
}

function normalizeArAction(raw: unknown, fallbackText = ''): ArAction {
  const s = `${String(raw ?? '')} ${fallbackText}`.toLowerCase();
  if (s.includes('turn_left') || s.includes('ccw') || s.includes('counter')) return 'turn_left';
  if (s.includes('turn_right') || s.includes('cw') || s.includes('clockwise')) return 'turn_right';
  if (s.includes('press') || s.includes('push') || s.includes('click')) return 'press';
  if (s.includes('pull') || s.includes('remove') || s.includes('lift')) return 'pull';
  if (s.includes('caution') || s.includes('danger') || s.includes('warning') || s.includes('voltage')) return 'caution';
  return 'point';
}

function normalizeArColor(raw: unknown): ArColor | undefined {
  const s = String(raw ?? '').toLowerCase();
  if (s.includes('orange') || s.includes('accent')) return 'orange';
  if (s.includes('green')) return 'green';
  if (s.includes('amber') || s.includes('yellow')) return 'amber';
  if (s.includes('red')) return 'red';
  if (s.includes('white') || s.includes('silver')) return 'white';
  return undefined;
}

async function handleToolCalls(message: LiveServerMessage): Promise<void> {
  const calls = message.toolCall?.functionCalls;
  if (!calls || !liveSession) return;

  const functionResponses = await Promise.all(
    calls.map(async (call) => {
      const args = (call.args ?? {}) as Record<string, unknown>;
      let result: Record<string, unknown> = { ok: true };

      if (call.name === 'set_task_plan') {
        hud.setTaskPlan(
          String(args.title ?? 'PROCEDURE'),
          Array.isArray(args.tools) ? args.tools.map(String) : [],
          Array.isArray(args.steps) ? args.steps.map(String) : []
        );
        result = { ok: true, message: 'Procedure displayed on HUD.' };
      } else if (call.name === 'update_step') {
        const progress = hud.updateStep(Number(args.step_number ?? 1), Boolean(args.done ?? true));
        if (progress.allDone) {
          gpuVision.stop();
          arOverlay.clear();
          btnYoloeLock?.classList.add('hidden');
        }
        result = { ok: true, ...progress };
      } else if (call.name === 'point_to_target') {
        const targetEn = String(args.target_en ?? args.label ?? 'object').trim();
        const label = String(args.label ?? targetEn);
        const action = normalizeArAction(args.action, label);
        const color = normalizeArColor(args.color) ?? 'orange';
        const stepTag = hud.getActiveStepTag();
        const displayLabel = stepTag ? `${stepTag} / ${label}` : label;

        gpuMissStreak = 0;
        btnYoloeLock?.classList.add('hidden');
        arOverlay.addTarget(
          {
            x: 500,
            y: 500,
            label: displayLabel,
            action,
            color,
            boxWidth: 100,
            boxHeight: 100,
            trackingState: 'relocalizing',
          },
          true
        );

        updateGpuBadge('LOAD YOLOE-11M', 'busy');
        const started = await gpuVision.trackText(targetEn);
        sendDiag('gpu_track_text', { targetEn, label, started });

        if (started) {
          hud.appendTranscript('system', `GPU YOLOE-11m-seg tracking active: "${targetEn}".`);
          hud.setStatus('connected', `Tracking: ${label}`);
          result = { ok: true, message: `Local GPU tracking started for "${targetEn}".` };
        } else {
          arOverlay.clear();
          updateGpuBadge('GPU OFFLINE', 'off');
          hud.appendTranscript('system', 'GPU Vision server (port 8000) is offline. Run npm run dev to start it.');
          result = { ok: false, message: 'Local GPU Vision server unavailable.' };
        }
      }

      return {
        id: call.id,
        name: call.name,
        response: result,
      };
    })
  );

  liveSession?.sendToolResponse({ functionResponses });
}

async function connectLive(): Promise<void> {
  audio.prepare();
  manualDisconnect = false;
  isSessionReady = false;
  txPackets = 0;
  rxAudioChunks = 0;
  peakRms = 0;
  sendDiag('connect_start', {
    url: window.location.href,
    secure: window.isSecureContext,
    gpu: gpuVision.gpuName,
  });
  hud.setStatus('idle', 'Connecting');
  hud.appendTranscript('system', 'Requesting ephemeral session token...');
  btnConnect.disabled = true;

  try {
    const res = await fetch('/api/token', { method: 'POST' });
    const data = (await res.json()) as { token?: string; model?: string; error?: string };
    if (!res.ok || !data.token) {
      throw new Error(data.error || `Token request failed (HTTP ${res.status})`);
    }

    hud.appendTranscript('system', 'Opening Gemini 3.8 Live channel...');
    const ai = new GoogleGenAI({
      apiKey: data.token,
      httpOptions: { apiVersion: 'v1beta' },
    });

    let resolveSetup!: () => void;
    const setupReadyPromise = new Promise<void>((resolve) => {
      resolveSetup = resolve;
    });

    liveSession = await ai.live.connect({
      model: data.model || 'gemini-3.8-live',
      config: {
        responseModalities: [Modality.AUDIO],
        systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
        tools: [{ functionDeclarations: HUD_TOOLS }],
        inputAudioTranscription: {},
        outputAudioTranscription: {},
      },
      callbacks: {
        onopen: () => {
          hud.setStatus('idle', 'Handshake');
        },
        onmessage: (message: LiveServerMessage) => {
          if (message.setupComplete) {
            isSessionReady = true;
            sendDiag('setup_complete');
            hud.setStatus('connected', 'Live');
            resolveSetup();
          }

          if (message.toolCall) {
            sendDiag('tool_call', { names: message.toolCall.functionCalls?.map((c) => c.name) });
            void handleToolCalls(message);
          }

          const content = message.serverContent;
          if (!content) return;

          if (content.interrupted) {
            sendDiag('interrupted');
            audio.clearPlayback();
            hud.setStatus('connected', 'Live');
          }

          if (content.modelTurn?.parts) {
            for (const part of content.modelTurn.parts) {
              if (part.inlineData?.data) {
                rxAudioChunks++;
                if (rxAudioChunks === 1 || rxAudioChunks % 20 === 0) {
                  sendDiag('rx_audio', { rxAudioChunks });
                }
                hud.setStatus('speaking', 'Speaking');
                audio.playPcm24k(part.inlineData.data);
              }
            }
          }

          if (content.inputTranscription?.text) {
            sendDiag('in_transcript', { text: content.inputTranscription.text });
            hud.appendTranscript('user', content.inputTranscription.text);
          }
          if (content.outputTranscription?.text) {
            sendDiag('out_transcript', { text: content.outputTranscription.text });
            hud.appendTranscript('model', content.outputTranscription.text);
          }
          if (content.turnComplete) {
            sendDiag('turn_complete', { txPackets, rxAudioChunks });
            hud.setStatus('connected', gpuVision.isTracking ? 'Live / Tracking' : 'Live');
          }
        },
        onerror: (e: ErrorEvent) => {
          console.error('Live API error:', e);
          const msg = `WebSocket error: ${e.message || 'Unknown'}`;
          sendDiag('ws_error', { msg });
          hud.setStatus('error', 'Link Error');
          hud.appendTranscript('system', msg);
        },
        onclose: (e: CloseEvent) => {
          console.warn('Live API closed:', e.code, e.reason);
          sendDiag('ws_close', { code: e.code, reason: e.reason, manualDisconnect });
          const reason = !manualDisconnect
            ? `Session closed (${e.code}${e.reason ? `: ${e.reason}` : ''})`
            : null;
          if (reason) hud.appendTranscript('system', reason);
          disconnectLive(reason);
        },
      },
    });

    await Promise.race([
      setupReadyPromise,
      new Promise<void>((resolve) => setTimeout(resolve, 1500)),
    ]);
    isSessionReady = true;
    hud.appendTranscript('system', 'Gemini 3.8 Live ready. Speak or type below.');

    await startMicWithSelectedDevice();

    // Stream 1 FPS video frames to Gemini 3.8 Live
    frameTimer = window.setInterval(() => {
      if (!isSessionReady || !liveSession) return;
      const jpeg = captureJpegBase64(768, 0.72);
      if (jpeg) {
        liveSession.sendRealtimeInput({
          video: { data: jpeg, mimeType: 'image/jpeg' },
        });
      }
    }, 1000);

    setConnectButton(true);
    textInput.disabled = false;
    btnSend.disabled = false;
    hud.setStatus('connected', 'Live');
  } catch (err) {
    console.error(err);
    const msg = err instanceof Error ? err.message : 'Connection failed';
    hud.appendTranscript('system', `Error: ${msg}`);
    disconnectLive(msg);
  } finally {
    btnConnect.disabled = false;
  }
}

function disconnectLive(errorMessage: string | null = null): void {
  isSessionReady = false;
  if (frameTimer !== null) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
  gpuVision.stop();
  audio.stop();
  arOverlay.clear();
  btnYoloeLock?.classList.add('hidden');
  if (micLevelBar) micLevelBar.style.width = '0%';
  txPackets = 0;
  if (txCounterEl) txCounterEl.textContent = 'TX 0';

  const sess = liveSession;
  liveSession = null;
  if (sess) {
    try {
      sess.close();
    } catch {
      // ignore
    }
  }

  setConnectButton(false);
  btnMute.disabled = true;
  audio.muted = false;
  setMuteButton(false);
  textInput.disabled = true;
  btnSend.disabled = true;

  if (errorMessage) {
    hud.setStatus('error', errorMessage);
  } else {
    hud.setStatus('idle', 'Standby');
  }
}

async function refreshMicList(selectedId?: string): Promise<void> {
  if (!micSelect) return;
  const mics = await audio.listMicrophones();
  const current = selectedId ?? micSelect.value;
  micSelect.innerHTML = '<option value="">Default Mic</option>';
  mics.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = m.deviceId;
    opt.textContent = m.label || `Microphone ${i + 1}`;
    if (m.deviceId && m.deviceId === current) opt.selected = true;
    micSelect.appendChild(opt);
  });
}

async function startMicWithSelectedDevice(): Promise<void> {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Browser microphone API unavailable (use http://localhost:3000)');
    }
    const deviceId = micSelect?.value || undefined;
    const label = await audio.startCapture(
      (pcmBase64) => {
        if (!isSessionReady || !liveSession) return;
        liveSession.sendRealtimeInput({
          audio: { data: pcmBase64, mimeType: 'audio/pcm;rate=16000' },
        });
        txPackets++;
        if (txCounterEl) {
          txCounterEl.textContent = `TX ${txPackets}`;
        }
        if (txPackets === 1 || txPackets % 30 === 0) {
          sendDiag('tx_heartbeat', { txPackets, peakRms: Number(peakRms.toFixed(4)) });
          peakRms = 0;
        }
      },
      (rms) => {
        if (rms > peakRms) peakRms = rms;
        if (micLevelBar) {
          const pct = Math.min(100, Math.round(rms * 550));
          micLevelBar.style.width = `${pct}%`;
        }
      },
      deviceId
    );
    await refreshMicList(deviceId);
    sendDiag('mic_active', { label, sampleRate: audio.getSampleRate() });
    hud.appendTranscript('system', `Mic active: ${label} (${audio.getSampleRate()} Hz -> 16 kHz)`);
    btnMute.disabled = false;
  } catch (micErr) {
    const micMsg = micErr instanceof Error ? micErr.message : String(micErr);
    sendDiag('mic_error', { micMsg });
    console.warn('Mic start failed:', micErr);
    hud.appendTranscript('system', `Microphone unavailable (${micMsg}). Use text input below.`);
    hud.showWarning(`Microphone unavailable: ${micMsg}`, 'caution');
  }
}

// Click any part or object in viewport -> SAM 2.1 Visual Memory Bank tracking
videoEl.addEventListener('click', (e) => {
  const { nx, ny } = arOverlay.screenToNorm(e.clientX, e.clientY);
  gpuMissStreak = 0;
  btnYoloeLock?.classList.add('hidden');
  arOverlay.addTarget(
    {
      x: nx,
      y: ny,
      label: 'SAM 2.1 Memory',
      action: 'point',
      color: 'orange',
      boxWidth: 70,
      boxHeight: 70,
      trackingState: 'relocalizing',
    },
    true
  );
  updateGpuBadge('LOAD SAM2-MEM', 'busy');
  void gpuVision.trackPoint(nx, ny).then((ok) => {
    if (ok) {
      sendDiag('gpu_track_point', { nx: Math.round(nx), ny: Math.round(ny) });
      hud.appendTranscript('system', 'SAM 2.1 Visual Memory locked onto selected target.');
    } else {
      arOverlay.clear();
      updateGpuBadge('GPU OFFLINE', 'off');
      hud.appendTranscript('system', 'GPU Vision server (port 8000) is offline.');
    }
  });
});

// Direct YOLOE-11m-seg open-vocabulary tracking bar
yoloeForm?.addEventListener('submit', (e) => {
  e.preventDefault();
  const prompt = yoloeInput?.value.trim() ?? '';
  if (!prompt) return;
  gpuMissStreak = 0;
  btnYoloeLock?.classList.add('hidden');
  arOverlay.addTarget(
    {
      x: 500,
      y: 500,
      label: prompt,
      action: 'point',
      color: 'green',
      boxWidth: 100,
      boxHeight: 100,
      trackingState: 'relocalizing',
    },
    true
  );
  updateGpuBadge('LOAD YOLOE-11M', 'busy');
  void gpuVision.trackText(prompt).then((ok) => {
    sendDiag('gpu_track_text_manual', { prompt, ok });
    if (ok) {
      hud.appendTranscript('system', `YOLOE-11m-seg scanning for "${prompt}". Use Lock SAM2 once detected.`);
    } else {
      arOverlay.clear();
      updateGpuBadge('GPU OFFLINE', 'off');
      hud.appendTranscript('system', 'GPU Vision server (port 8000) is offline.');
    }
  });
});

btnYoloeLock?.addEventListener('click', () => {
  const ok = gpuVision.lockCurrentToSam();
  if (ok) {
    btnYoloeLock.classList.add('hidden');
    sendDiag('gpu_lock_to_sam');
    hud.appendTranscript('system', 'Target handed off from YOLOE to SAM 2.1 Visual Memory.');
  }
});

micSelect?.addEventListener('change', () => {
  if (liveSession && isSessionReady) {
    void startMicWithSelectedDevice();
  }
});

btnConnect.addEventListener('click', () => {
  if (liveSession) {
    manualDisconnect = true;
    disconnectLive();
  } else {
    void connectLive();
  }
});

btnTestAudio?.addEventListener('click', () => {
  const state = audio.playTestTone();
  hud.appendTranscript('system', `Audio test tone played (AudioContext: ${state}).`);
});

btnMute.addEventListener('click', () => {
  audio.muted = !audio.muted;
  setMuteButton(audio.muted);
  if (audio.muted) {
    liveSession?.sendRealtimeInput({ audioStreamEnd: true });
  }
});

btnSwitchCam.addEventListener('click', () => {
  gpuVision.stop();
  arOverlay.clear();
  btnYoloeLock?.classList.add('hidden');
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  void startCamera();
});

textForm.addEventListener('submit', (e) => {
  e.preventDefault();
  audio.prepare();
  const text = textInput.value.trim();
  if (!text || !liveSession) return;
  sendDiag('text_send', { text });
  hud.appendTranscript('user', `${text}\n`);
  liveSession.sendClientContent({
    turns: [{ role: 'user', parts: [{ text }] }],
    turnComplete: true,
  });
  textInput.value = '';
});

void startCamera().catch((err) => {
  const msg = err instanceof Error ? err.message : 'Camera initialization failed';
  hud.setStatus('idle', 'No Camera');
  hud.appendTranscript('system', `Camera unavailable: ${msg}`);
});

void gpuVision.checkHealth().then((h) => {
  if (h.ok) {
    updateGpuBadge((h.gpu || 'CUDA').replace('NVIDIA GeForce ', '').toUpperCase(), 'ready');
    hud.appendTranscript(
      'system',
      `Local GPU Vision online (${h.gpu} / YOLOE-11m-seg + SAM 2.1 Memory).`
    );
  } else {
    updateGpuBadge('GPU OFFLINE', 'off');
    hud.appendTranscript(
      'system',
      'Local GPU Vision server (port 8000) offline. Start with npm run dev.'
    );
  }
});
