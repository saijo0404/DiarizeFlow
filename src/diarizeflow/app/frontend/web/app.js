// DiarizeFlow Web Floating Overlay & Audio Controller

const SPEAKER_COLORS = [
  "#38bdf8", // Sky Blue
  "#c084fc", // Purple
  "#34d399", // Emerald
  "#f472b6", // Pink
  "#fbbf24", // Amber
  "#60a5fa", // Blue
  "#a78bfa", // Violet
  "#f87171", // Rose
];

function getSpeakerColor(speakerLabel) {
  const match = speakerLabel.match(/\d+/);
  if (match) {
    const idx = (parseInt(match[0], 10) - 1) % SPEAKER_COLORS.length;
    return SPEAKER_COLORS[idx];
  }
  return SPEAKER_COLORS[0];
}

class DiarizeOverlayApp {
  constructor() {
    this.container = document.getElementById("subtitle-container");
    this.statusDot = document.getElementById("status-dot");
    this.statusText = document.getElementById("status-text");

    this.fadeDelay = 5.0; // seconds before fade-out
    this.wsSubtitles = null;
    this.wsAudio = null;

    this.audioContext = null;
    this.mediaStream = null;
    this.isRecording = false;

    this.initWebSockets();
    this.initUIEvents();
    this.loadConfig();
  }

  initWebSockets() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;

    // 1. Subtitles WebSocket
    const subUrl = `${protocol}//${host}/ws/subtitles`;
    this.wsSubtitles = new WebSocket(subUrl);

    this.wsSubtitles.onopen = () => {
      this.statusDot.classList.add("connected");
      this.statusText.textContent = "已連線";
    };

    this.wsSubtitles.onclose = () => {
      this.statusDot.classList.remove("connected");
      this.statusText.textContent = "連線中斷，重連中...";
      setTimeout(() => this.initWebSockets(), 2500);
    };

    this.wsSubtitles.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "connection") {
          console.log("[✓] Connected to DiarizeFlow server:", data);
        } else if (data.translated_text) {
          this.displaySubtitle(data);
        }
      } catch (err) {
        console.error("Failed to parse subtitle message:", err);
      }
    };

    // 2. Audio WebSocket (created on demand during recording)
    this.audioWsUrl = `${protocol}//${host}/ws/audio`;
  }

  displaySubtitle(data) {
    const speaker = data.speaker || "講者 1";
    const original = data.original_text || "";
    const translated = data.translated_text || "";
    const color = getSpeakerColor(speaker);

    // Limit maximum active subtitle cards to 2 to prevent screen clutter
    while (this.container.children.length >= 2) {
      this.container.removeChild(this.container.firstChild);
    }

    const card = document.createElement("div");
    card.className = "subtitle-card";

    card.innerHTML = `
      <div class="subtitle-header">
        <span class="speaker-badge" style="background: ${color}22; color: ${color}; border: 1px solid ${color}66;">
          ${speaker}
        </span>
        <span class="timestamp">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="translated-text">${translated}</div>
      ${original ? `<div class="original-text">${original}</div>` : ""}
    `;

    this.container.appendChild(card);

    // Auto-dismiss: Fade out and remove card after configured seconds
    setTimeout(() => {
      card.classList.add("fade-out");
      setTimeout(() => {
        if (card.parentNode) {
          card.parentNode.removeChild(card);
        }
      }, 500);
    }, this.fadeDelay * 1000);
  }

  async startBrowserAudioCapture(mode = "mic") {
    if (this.isRecording) {
      this.stopBrowserAudioCapture();
      return;
    }

    try {
      if (mode === "system") {
        // Capture system/tab audio via getDisplayMedia
        this.mediaStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });
      } else {
        // Capture microphone
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
      }

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Open Audio WebSocket
      this.wsAudio = new WebSocket(this.audioWsUrl);
      this.wsAudio.binaryType = "arraybuffer";

      this.wsAudio.onopen = () => {
        const bufferSize = 4096;
        const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

        processor.onaudioprocess = (e) => {
          if (!this.isRecording || this.wsAudio.readyState !== WebSocket.OPEN) return;
          const inputData = e.inputBuffer.getChannelData(0);
          // Send 16kHz float32 buffer directly
          this.wsAudio.send(inputData.buffer);
        };

        source.connect(processor);
        processor.connect(this.audioContext.destination);

        this.isRecording = true;
        this.updateCaptureButtonState(mode, true);
        console.log(`[✓] Started browser ${mode} audio capture.`);
      };
    } catch (err) {
      console.error("Audio capture failed:", err);
      alert(`無法啟動音訊捕捉: ${err.message}`);
    }
  }

  stopBrowserAudioCapture() {
    this.isRecording = false;
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    if (this.wsAudio) {
      this.wsAudio.close();
      this.wsAudio = null;
    }
    this.updateCaptureButtonState(null, false);
    console.log("[*] Stopped browser audio capture.");
  }

  updateCaptureButtonState(mode, active) {
    const btnMic = document.getElementById("btn-web-mic");
    const btnSys = document.getElementById("btn-web-system");

    if (!active) {
      btnMic.textContent = "🎤 瀏覽器收音";
      btnMic.className = "btn btn-primary";
      btnSys.textContent = "🖥️ 系統聲音 (標籤頁/遊戲)";
      btnSys.className = "btn";
    } else if (mode === "mic") {
      btnMic.textContent = "⏹ 停止麥克風";
      btnMic.className = "btn btn-danger";
    } else if (mode === "system") {
      btnSys.textContent = "⏹ 停止系統聲音";
      btnSys.className = "btn btn-danger";
    }
  }

  initUIEvents() {
    document.getElementById("btn-web-mic").addEventListener("click", () => {
      this.startBrowserAudioCapture("mic");
    });

    document.getElementById("btn-web-system").addEventListener("click", () => {
      this.startBrowserAudioCapture("system");
    });

    const modal = document.getElementById("settings-modal");
    document.getElementById("btn-settings").addEventListener("click", () => {
      modal.classList.add("active");
    });

    const closeModal = () => modal.classList.remove("active");
    document.getElementById("btn-modal-close").addEventListener("click", closeModal);
    document.getElementById("btn-cancel-settings").addEventListener("click", closeModal);

    document.getElementById("btn-save-settings").addEventListener("click", () => {
      this.saveConfig();
      closeModal();
    });

    // Test audio file upload
    const testAudioInput = document.getElementById("setting-test-audio");
    testAudioInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      try {
        const resp = await fetch("/api/test/audio", { method: "POST", body: formData });
        const res = await resp.json();
        alert(`音訊已送出測試 (${res.duration} 秒)，即將顯示分離語者與翻譯結果！`);
        closeModal();
      } catch (err) {
        alert("音訊測試上傳失敗: " + err);
      }
    });
  }

  async loadConfig() {
    try {
      const resp = await fetch("/api/config");
      const cfg = await resp.json();
      if (cfg.llm) {
        document.getElementById("setting-target-lang").value = cfg.llm.target_language || "繁體中文";
        document.getElementById("setting-provider").value = cfg.llm.provider || "vllm";
        document.getElementById("setting-base-url").value = cfg.llm.base_url || "http://127.0.0.1:8000/v1";
        document.getElementById("setting-model").value = cfg.llm.model_name || "";
      }
      if (cfg.ui) {
        this.fadeDelay = cfg.ui.fade_out_seconds || 5.0;
        document.getElementById("setting-fade-delay").value = this.fadeDelay;
      }
    } catch (err) {
      console.warn("Failed to load server config:", err);
    }
  }

  async saveConfig() {
    const targetLang = document.getElementById("setting-target-lang").value;
    const provider = document.getElementById("setting-provider").value;
    const baseUrl = document.getElementById("setting-base-url").value;
    const model = document.getElementById("setting-model").value;
    const fadeDelay = parseFloat(document.getElementById("setting-fade-delay").value) || 5.0;
    this.fadeDelay = fadeDelay;

    const payload = {
      llm: {
        target_language: targetLang,
        provider: provider,
        base_url: baseUrl,
        model_name: model,
      },
      ui: {
        fade_out_seconds: fadeDelay,
      },
    };

    try {
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      console.log("[✓] Configuration saved.");
    } catch (err) {
      console.error("Failed to save config:", err);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new DiarizeOverlayApp();
});
