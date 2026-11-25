export interface HudStep {
  text: string;
  done: boolean;
}

export class HudController {
  private statusDot = document.getElementById('status-dot')!;
  private statusText = document.getElementById('status-text')!;
  private warningBanner = document.getElementById('warning-banner')!;
  private warningText = document.getElementById('warning-text')!;
  private toolsPanel = document.getElementById('tools-panel')!;
  private toolsList = document.getElementById('tools-list')!;
  private toolsCount = document.getElementById('tools-count')!;
  private stepsPanel = document.getElementById('steps-panel')!;
  private taskTitle = document.getElementById('task-title')!;
  private stepsList = document.getElementById('steps-list')!;
  private stepsProgress = document.getElementById('steps-progress')!;
  private transcriptBox = document.getElementById('transcript-box')!;

  private steps: HudStep[] = [];
  private activeIndex = 0;

  constructor() {
    document.getElementById('btn-dismiss-warning')?.addEventListener('click', () => {
      this.clearWarning();
    });
  }

  setStatus(state: 'idle' | 'connected' | 'speaking' | 'error', label: string): void {
    this.statusDot.className = `status-led ${state === 'idle' ? '' : state}`;
    this.statusText.textContent = label;
  }

  setTaskPlan(title: string, tools: string[], stepTexts: string[]): void {
    this.taskTitle.textContent = title || 'PROCEDURE';

    if (tools.length > 0) {
      this.toolsPanel.classList.remove('hidden');
      this.toolsCount.textContent = String(tools.length).padStart(2, '0');
      this.toolsList.innerHTML = '';
      for (const item of tools) {
        const li = document.createElement('li');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        const span = document.createElement('span');
        span.textContent = item;
        li.append(cb, span);
        this.toolsList.appendChild(li);
      }
    }

    this.steps = stepTexts.map((text) => ({ text, done: false }));
    this.activeIndex = 0;
    this.renderSteps();
  }

  updateStep(stepNumber: number, done: boolean): { doneCount: number; total: number; allDone: boolean } {
    const idx = stepNumber - 1;
    if (idx >= 0 && idx < this.steps.length) {
      this.steps[idx].done = done;
      const nextUndone = this.steps.findIndex((s) => !s.done);
      this.activeIndex = nextUndone === -1 ? this.steps.length : nextUndone;
      this.renderSteps();
    }
    const doneCount = this.steps.filter((s) => s.done).length;
    return {
      doneCount,
      total: this.steps.length,
      allDone: this.steps.length > 0 && doneCount === this.steps.length,
    };
  }

  getActiveStepTag(): string {
    if (this.steps.length === 0 || this.activeIndex >= this.steps.length) return '';
    return `STEP ${this.activeIndex + 1}/${this.steps.length}`;
  }

  private renderSteps(): void {
    if (this.steps.length === 0) {
      this.stepsPanel.classList.add('hidden');
      return;
    }
    this.stepsPanel.classList.remove('hidden');
    const doneCount = this.steps.filter((s) => s.done).length;
    this.stepsProgress.textContent = `${doneCount}/${this.steps.length}`;
    this.stepsList.innerHTML = '';

    this.steps.forEach((step, idx) => {
      const li = document.createElement('li');
      if (step.done) li.classList.add('done-step');
      else if (idx === this.activeIndex) li.classList.add('active-step');

      const indexEl = document.createElement('span');
      indexEl.className = 'step-index';
      indexEl.textContent = step.done ? 'OK' : String(idx + 1).padStart(2, '0');

      const bodyEl = document.createElement('span');
      bodyEl.textContent = step.text;

      li.append(indexEl, bodyEl);
      this.stepsList.appendChild(li);
    });
  }

  showWarning(message: string, severity: 'danger' | 'caution' = 'danger'): void {
    this.warningText.textContent = message;
    this.warningBanner.classList.remove('hidden', 'caution');
    if (severity === 'caution') this.warningBanner.classList.add('caution');
  }

  clearWarning(): void {
    this.warningBanner.classList.add('hidden');
  }

  appendTranscript(role: 'user' | 'model' | 'system', text: string): void {
    if (!text.trim()) return;
    const hint = this.transcriptBox.querySelector('.transcript-hint');
    if (hint) hint.remove();

    const last = this.transcriptBox.lastElementChild as HTMLElement | null;
    if (role !== 'system' && last && last.dataset.role === role) {
      last.querySelector('.transcript-body')!.textContent += text;
    } else {
      const div = document.createElement('div');
      div.className = `transcript-line ${role}`;
      div.dataset.role = role;

      const roleEl = document.createElement('span');
      roleEl.className = 'transcript-role';
      roleEl.textContent = role === 'user' ? 'YOU' : role === 'model' ? 'GUIDE' : 'SYS';

      const bodyEl = document.createElement('span');
      bodyEl.className = 'transcript-body';
      bodyEl.textContent = text;

      div.append(roleEl, bodyEl);
      this.transcriptBox.appendChild(div);
    }
    this.transcriptBox.scrollTop = this.transcriptBox.scrollHeight;
  }
}
