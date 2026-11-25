/**
 * Clean Audio pipeline for Gemini Live API (matching official Google AI Studio pattern):
 * - Capture: Pure, unmodified mic audio (browser WebRTC AEC/NS/AGC only)
 *            -> AudioWorkletNode (2400 samples @ 48kHz = exact 50ms)
 *            -> exact 1600-sample (100ms @ 16kHz = 3200 bytes) 16-bit LE PCM
 * - Playback: 24kHz, mono, 16-bit LE PCM with instant barge-in clear
 */

const WORKLET_CODE = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._bufferLength = 0;
    this._targetSize = 2400; // 50ms @ 48kHz -> clean 3:1 downsample to 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;
    const channel = input[0];
    const copy = new Float32Array(channel.length);
    copy.set(channel);
    this._buffer.push(copy);
    this._bufferLength += copy.length;

    if (this._bufferLength >= this._targetSize) {
      const merged = new Float32Array(this._bufferLength);
      let offset = 0;
      for (const chunk of this._buffer) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      this._buffer = [];
      this._bufferLength = 0;
      this.port.postMessage(merged, [merged.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-capture-processor', PcmCaptureProcessor);
`;

function floatTo16BitPCM(float32: Float32Array): Int16Array {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function downsampleTo16k(buffer: Float32Array, inputSampleRate: number): Float32Array {
  if (inputSampleRate === 16000) return buffer;
  const ratio = inputSampleRate / 16000;
  const newLength = Math.max(1, Math.round(buffer.length / ratio));
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function base64ToInt16(base64: string): Int16Array {
  const binary = atob(base64);
  const byteLen = binary.length - (binary.length % 2);
  const bytes = new Uint8Array(byteLen);
  for (let i = 0; i < byteLen; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

export class AudioStreamer {
  private captureCtx: AudioContext | null = null;
  private playbackCtx: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private workletLoadedCtx: AudioContext | null = null;
  private pcmQueue: number[] = [];
  private nextPlayTime = 0;
  private activeSources: AudioBufferSourceNode[] = [];

  public muted = false;
  public activeMicLabel = '';

  prepare(): void {
    if (!this.captureCtx || this.captureCtx.state === 'closed') {
      this.captureCtx = new AudioContext();
    }
    if (this.captureCtx.state === 'suspended') {
      void this.captureCtx.resume();
    }

    if (!this.playbackCtx || this.playbackCtx.state === 'closed') {
      this.playbackCtx = new AudioContext({ sampleRate: 24000 });
    }
    if (this.playbackCtx.state === 'suspended') {
      void this.playbackCtx.resume();
    }
  }

  playTestTone(): string {
    this.prepare();
    const ctx = this.playbackCtx!;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(660, ctx.currentTime);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.25);
    return ctx.state;
  }

  getSampleRate(): number {
    return this.captureCtx?.sampleRate ?? 0;
  }

  async listMicrophones(): Promise<MediaDeviceInfo[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'audioinput');
  }

  stopCapture(): void {
    this.pcmQueue = [];
    this.workletNode?.port.close();
    this.workletNode?.disconnect();
    this.workletNode = null;
    this.micSource?.disconnect();
    this.micSource = null;
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micStream = null;
  }

  async startCapture(
    onPcmBase64: (base64: string) => void,
    onMicLevel?: (rms: number) => void,
    deviceId?: string
  ): Promise<string> {
    this.prepare();
    this.stopCapture();

    const audioConstraints: MediaTrackConstraints = {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
    };

    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
    });

    const track = this.micStream.getAudioTracks()[0];
    this.activeMicLabel = track?.label || 'Default Microphone';

    if (this.captureCtx!.state === 'suspended') {
      await this.captureCtx!.resume();
    }

    if (this.workletLoadedCtx !== this.captureCtx) {
      const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
      const workletUrl = URL.createObjectURL(blob);
      try {
        await this.captureCtx!.audioWorklet.addModule(workletUrl);
        this.workletLoadedCtx = this.captureCtx;
      } finally {
        URL.revokeObjectURL(workletUrl);
      }
    }

    this.micSource = this.captureCtx!.createMediaStreamSource(this.micStream);
    this.workletNode = new AudioWorkletNode(this.captureCtx!, 'pcm-capture-processor');

    this.workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
      if (this.muted) {
        onMicLevel?.(0);
        return;
      }
      const input = event.data;
      if (!input || input.length === 0) return;

      let sumSq = 0;
      for (let i = 0; i < input.length; i++) {
        sumSq += input[i] * input[i];
      }
      onMicLevel?.(Math.sqrt(sumSq / input.length));

      // Pure, unmodified 16kHz PCM in exact 1600-sample (100ms) frames
      const resampled = downsampleTo16k(input, this.captureCtx!.sampleRate);
      const pcm16 = floatTo16BitPCM(resampled);

      for (let i = 0; i < pcm16.length; i++) {
        this.pcmQueue.push(pcm16[i]);
      }
      const CHUNK_SAMPLES = 1600;
      while (this.pcmQueue.length >= CHUNK_SAMPLES) {
        const chunk = new Int16Array(this.pcmQueue.splice(0, CHUNK_SAMPLES));
        const base64 = bytesToBase64(new Uint8Array(chunk.buffer));
        onPcmBase64(base64);
      }
    };

    this.micSource.connect(this.workletNode);
    this.workletNode.connect(this.captureCtx!.destination);
    return this.activeMicLabel;
  }

  playPcm24k(base64Audio: string): void {
    if (!this.playbackCtx || this.playbackCtx.state === 'closed') {
      this.playbackCtx = new AudioContext({ sampleRate: 24000 });
    }
    if (this.playbackCtx.state === 'suspended') {
      void this.playbackCtx.resume();
    }

    const int16 = base64ToInt16(base64Audio);
    if (int16.length === 0) return;

    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 0x8000;
    }

    const buffer = this.playbackCtx.createBuffer(1, float32.length, 24000);
    buffer.getChannelData(0).set(float32);

    const source = this.playbackCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.playbackCtx.destination);

    const now = this.playbackCtx.currentTime;
    const startTime = this.nextPlayTime < now ? now + 0.02 : this.nextPlayTime;
    source.start(startTime);
    this.nextPlayTime = startTime + buffer.duration;

    this.activeSources.push(source);
    source.onended = () => {
      this.activeSources = this.activeSources.filter((s) => s !== source);
    };
  }

  clearPlayback(): void {
    for (const src of this.activeSources) {
      try {
        src.stop();
      } catch {
        // already stopped
      }
    }
    this.activeSources = [];
    this.nextPlayTime = 0;
  }

  stop(): void {
    this.clearPlayback();
    this.stopCapture();
    void this.captureCtx?.close();
    this.captureCtx = null;
    this.workletLoadedCtx = null;
    void this.playbackCtx?.close();
    this.playbackCtx = null;
  }
}
