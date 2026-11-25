/**
 * AR Spatial Overlay Controller — Precision Optical Instrument Style
 * - Exact bidirectional mapping between 0..1000 video space and `object-fit: cover` screen space
 * - Crisp non-blurred SVG pixel-silhouette polygons, optical corner ticks, and technical callouts
 * - Zero emoji icons, zero neon glow slop
 */

export type ArAction = 'point' | 'turn_left' | 'turn_right' | 'press' | 'pull' | 'caution';
export type ArColor = 'orange' | 'green' | 'amber' | 'red' | 'white';

export interface ArTargetMarker {
  x: number; // 0..1000
  y: number; // 0..1000
  label: string;
  action: ArAction;
  color?: ArColor;
  boxWidth?: number; // 0..1000
  boxHeight?: number; // 0..1000
  polygon?: Array<[number, number]>; // [[x, y], ...] in 0..1000
  trackingState?: 'locked' | 'relocalizing' | 'static';
  confidence?: number;
}

export interface ArArrowMarker {
  fromX: number; // 0..1000
  fromY: number; // 0..1000
  toX: number; // 0..1000
  toY: number; // 0..1000
  label: string;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

const COLOR_HEX: Record<ArColor, string> = {
  orange: '#ff5a1f',
  green: '#22c55e',
  amber: '#eab308',
  red: '#ef4444',
  white: '#eceef0',
};

function clampNorm(v: number): number {
  if (!Number.isFinite(v)) return 500;
  return Math.max(0, Math.min(1000, v));
}

function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {}
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    node.setAttribute(k, String(v));
  }
  return node;
}

export class ArOverlayController {
  private svg: SVGSVGElement;
  private labelsLayer: HTMLDivElement;
  private video: HTMLVideoElement;
  private clearBtn: HTMLButtonElement | null;
  private onClearCallback?: () => void;

  private targets: ArTargetMarker[] = [];
  private arrows: ArArrowMarker[] = [];

  constructor(
    svg: SVGSVGElement,
    labelsLayer: HTMLDivElement,
    video: HTMLVideoElement,
    clearBtn: HTMLButtonElement | null,
    onClear?: () => void
  ) {
    this.svg = svg;
    this.labelsLayer = labelsLayer;
    this.video = video;
    this.clearBtn = clearBtn;
    this.onClearCallback = onClear;

    this.clearBtn?.addEventListener('click', () => {
      this.clear();
      this.onClearCallback?.();
    });
    window.addEventListener('resize', () => this.render());
    this.video.addEventListener('loadedmetadata', () => this.render());
  }

  private normToScreen(nx: number, ny: number): { px: number; py: number } {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const vidW = this.video.videoWidth || vw;
    const vidH = this.video.videoHeight || vh;

    const scale = Math.max(vw / vidW, vh / vidH);
    const renderedW = vidW * scale;
    const renderedH = vidH * scale;
    const offsetX = (vw - renderedW) / 2;
    const offsetY = (vh - renderedH) / 2;

    return {
      px: offsetX + (clampNorm(nx) / 1000) * renderedW,
      py: offsetY + (clampNorm(ny) / 1000) * renderedH,
    };
  }

  screenToNorm(clientX: number, clientY: number): { nx: number; ny: number } {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const vidW = this.video.videoWidth || vw;
    const vidH = this.video.videoHeight || vh;

    const scale = Math.max(vw / vidW, vh / vidH);
    const renderedW = vidW * scale;
    const renderedH = vidH * scale;
    const offsetX = (vw - renderedW) / 2;
    const offsetY = (vh - renderedH) / 2;

    const nx = ((clientX - offsetX) / renderedW) * 1000;
    const ny = ((clientY - offsetY) / renderedH) * 1000;
    return { nx: clampNorm(nx), ny: clampNorm(ny) };
  }

  private normSizeToScreen(nw: number, nh: number): { pw: number; ph: number } {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const vidW = this.video.videoWidth || vw;
    const vidH = this.video.videoHeight || vh;
    const scale = Math.max(vw / vidW, vh / vidH);
    return {
      pw: (clampNorm(nw) / 1000) * (vidW * scale),
      ph: (clampNorm(nh) / 1000) * (vidH * scale),
    };
  }

  addTarget(marker: ArTargetMarker, clearPrevious = true): void {
    if (clearPrevious) {
      this.targets = [];
      this.arrows = [];
    }
    this.targets.push({
      x: clampNorm(marker.x),
      y: clampNorm(marker.y),
      label: marker.label,
      action: marker.action,
      color: marker.color,
      boxWidth: marker.boxWidth,
      boxHeight: marker.boxHeight,
      polygon: marker.polygon,
      trackingState: marker.trackingState ?? 'static',
      confidence: marker.confidence,
    });
    this.render();
  }

  updatePrimaryTarget(update: {
    x: number;
    y: number;
    boxWidth: number;
    boxHeight: number;
    polygon?: Array<[number, number]>;
    trackingState: 'locked' | 'relocalizing' | 'static';
    confidence?: number;
  }): void {
    if (this.targets.length === 0) return;
    const t = this.targets[this.targets.length - 1];
    t.x = clampNorm(update.x);
    t.y = clampNorm(update.y);
    t.boxWidth = update.boxWidth;
    t.boxHeight = update.boxHeight;
    if (update.polygon) t.polygon = update.polygon;
    t.trackingState = update.trackingState;
    t.confidence = update.confidence;
    this.render(true);
  }

  addArrow(arrow: ArArrowMarker, clearPrevious = false): void {
    if (clearPrevious) {
      this.targets = [];
      this.arrows = [];
    }
    this.arrows.push({
      fromX: clampNorm(arrow.fromX),
      fromY: clampNorm(arrow.fromY),
      toX: clampNorm(arrow.toX),
      toY: clampNorm(arrow.toY),
      label: arrow.label,
    });
    this.render();
  }

  clear(): void {
    this.targets = [];
    this.arrows = [];
    this.render();
  }

  private getActionMeta(action: ArAction, customColor?: ArColor): { color: string; tag: string } {
    let defaultColor = COLOR_HEX.orange;
    let tag = 'TARGET';
    switch (action) {
      case 'turn_left':
        tag = 'TURN CCW';
        break;
      case 'turn_right':
        tag = 'TURN CW';
        break;
      case 'press':
        tag = 'PRESS';
        break;
      case 'pull':
        defaultColor = COLOR_HEX.amber;
        tag = 'PULL';
        break;
      case 'caution':
        defaultColor = COLOR_HEX.red;
        tag = 'CAUTION';
        break;
    }
    return {
      color: customColor && COLOR_HEX[customColor] ? COLOR_HEX[customColor] : defaultColor,
      tag,
    };
  }

  render(isTrackingTick = false): void {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    this.svg.setAttribute('viewBox', `0 0 ${vw} ${vh}`);
    this.svg.innerHTML = '';
    this.labelsLayer.innerHTML = '';

    const hasLocked = this.targets.length > 0 || this.arrows.length > 0;
    this.clearBtn?.classList.toggle('hidden', !hasLocked);
    if (!hasLocked) return;

    const defs = el('defs');
    const marker = el('marker', {
      id: 'ar-arrowhead',
      markerWidth: 10,
      markerHeight: 10,
      refX: 8,
      refY: 5,
      orient: 'auto',
    });
    marker.appendChild(el('path', { d: 'M1,1 L9,5 L1,9 L3,5 Z', fill: COLOR_HEX.orange }));
    defs.appendChild(marker);
    this.svg.appendChild(defs);

    // 1. Render Targets
    for (const t of this.targets) {
      const { px, py } = this.normToScreen(t.x, t.y);
      const { color, tag } = this.getActionMeta(t.action, t.color);
      const isScanning = t.trackingState === 'relocalizing';
      const strokeColor = isScanning ? COLOR_HEX.amber : color;
      const g = el('g', { opacity: isScanning ? 0.72 : 1 });

      // Crisp pixel-silhouette polygon mask (no blur/glow)
      if (t.polygon && t.polygon.length >= 3) {
        const pts = t.polygon
          .map(([vx, vy]) => {
            const p = this.normToScreen(vx, vy);
            return `${p.px.toFixed(1)},${p.py.toFixed(1)}`;
          })
          .join(' ');
        g.appendChild(
          el('polygon', {
            points: pts,
            fill: `${strokeColor}29`,
            stroke: strokeColor,
            'stroke-width': 1.75,
            'stroke-linejoin': 'miter',
          })
        );
      } else if (t.boxWidth && t.boxHeight && t.boxWidth > 12 && t.boxHeight > 12) {
        // Minimal optical corner brackets instead of a heavy solid box
        const { pw, ph } = this.normSizeToScreen(t.boxWidth, t.boxHeight);
        const x1 = px - pw / 2;
        const y1 = py - ph / 2;
        const x2 = px + pw / 2;
        const y2 = py + ph / 2;
        const tick = Math.max(8, Math.min(20, Math.min(pw, ph) * 0.28));
        const d = [
          `M ${x1} ${y1 + tick} L ${x1} ${y1} L ${x1 + tick} ${y1}`,
          `M ${x2 - tick} ${y1} L ${x2} ${y1} L ${x2} ${y1 + tick}`,
          `M ${x1} ${y2 - tick} L ${x1} ${y2} L ${x1 + tick} ${y2}`,
          `M ${x2 - tick} ${y2} L ${x2} ${y2} L ${x2} ${y2 - tick}`,
        ].join(' ');
        g.appendChild(
          el('path', {
            d,
            fill: 'none',
            stroke: strokeColor,
            'stroke-width': 1.75,
            'stroke-dasharray': isScanning ? '4 4' : 'none',
          })
        );
      }

      // Action-specific precision indicator
      if (t.action === 'turn_left' || t.action === 'turn_right') {
        g.appendChild(
          el('circle', {
            cx: px,
            cy: py,
            r: 26,
            fill: 'none',
            stroke: strokeColor,
            'stroke-width': 2,
            'stroke-dasharray': '30 22',
            class: t.action === 'turn_left' ? 'ar-spin-ccw' : 'ar-spin-cw',
            style: `transform-origin: ${px}px ${py}px`,
          })
        );
      } else if (t.action === 'pull') {
        g.appendChild(
          el('path', {
            d: `M ${px} ${py - 8} L ${px} ${py - 38} M ${px - 7} ${py - 30} L ${px} ${py - 40} L ${px + 7} ${py - 30}`,
            fill: 'none',
            stroke: strokeColor,
            'stroke-width': 2,
            class: 'ar-bounce-up',
          })
        );
      } else {
        // Precision crosshair reticle
        const r1 = 6;
        const r2 = 14;
        g.appendChild(
          el('path', {
            d: `M ${px - r2} ${py} L ${px - r1} ${py} M ${px + r1} ${py} L ${px + r2} ${py} M ${px} ${py - r2} L ${px} ${py - r1} M ${px} ${py + r1} L ${px} ${py + r2}`,
            stroke: strokeColor,
            'stroke-width': 1.5,
          })
        );
      }

      // Center precision pip
      g.appendChild(el('rect', { x: px - 2, y: py - 2, width: 4, height: 4, fill: strokeColor }));
      this.svg.appendChild(g);

      // Technical Callout Tag
      const callout = document.createElement('div');
      callout.className = isTrackingTick ? 'ar-callout no-anim' : 'ar-callout';
      callout.style.borderLeftColor = strokeColor;

      const labelX = Math.max(100, Math.min(vw - 100, px));
      const labelY = py > vh - 210 ? py - 44 : py + 24;
      callout.style.left = `${labelX}px`;
      callout.style.top = `${labelY}px`;

      const statusText = isScanning
        ? 'SCANNING'
        : t.confidence !== undefined
        ? `${tag} ${Math.round(t.confidence * 100)}%`
        : tag;

      const tagEl = document.createElement('span');
      tagEl.className = 'ar-callout-tag';
      tagEl.style.color = strokeColor;
      tagEl.textContent = statusText;

      const sepEl = document.createElement('span');
      sepEl.className = 'ar-callout-sep';

      const textEl = document.createElement('span');
      textEl.className = 'ar-callout-label';
      textEl.textContent = isScanning ? 'Searching viewport...' : t.label;

      callout.append(tagEl, sepEl, textEl);
      this.labelsLayer.appendChild(callout);
    }

    // 2. Render Point-to-Point Arrows
    for (const a of this.arrows) {
      const from = this.normToScreen(a.fromX, a.fromY);
      const to = this.normToScreen(a.toX, a.toY);
      const g = el('g');

      const midX = (from.px + to.px) / 2;
      const midY = (from.py + to.py) / 2 - 24;

      g.appendChild(el('rect', { x: from.px - 3, y: from.py - 3, width: 6, height: 6, fill: COLOR_HEX.orange }));
      g.appendChild(
        el('path', {
          d: `M ${from.px} ${from.py} Q ${midX} ${midY} ${to.px} ${to.py}`,
          fill: 'none',
          stroke: COLOR_HEX.orange,
          'stroke-width': 2.2,
          'stroke-dasharray': '8 4',
          'marker-end': 'url(#ar-arrowhead)',
          class: 'ar-dash-flow',
        })
      );
      this.svg.appendChild(g);

      if (a.label) {
        const callout = document.createElement('div');
        callout.className = 'ar-callout';
        callout.style.left = `${Math.max(100, Math.min(vw - 100, midX))}px`;
        callout.style.top = `${Math.max(76, Math.min(vh - 180, midY - 12))}px`;

        const tagEl = document.createElement('span');
        tagEl.className = 'ar-callout-tag';
        tagEl.style.color = COLOR_HEX.orange;
        tagEl.textContent = 'VECTOR';

        const sepEl = document.createElement('span');
        sepEl.className = 'ar-callout-sep';

        const textEl = document.createElement('span');
        textEl.className = 'ar-callout-label';
        textEl.textContent = a.label;

        callout.append(tagEl, sepEl, textEl);
        this.labelsLayer.appendChild(callout);
      }
    }
  }
}
