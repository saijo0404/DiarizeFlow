"""Settings dialog for audio devices, ASR engine, LLM translation, and UI appearance."""

import os
from pathlib import Path
import sys
import time
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diarizeflow.app.audio.devices import list_audio_devices
from diarizeflow.app.config import AppConfig, resolve_app_path


class SettingsDialog(QDialog):
    """Configuration dialog for audio devices, translation, and UI appearance."""

    settings_saved = Signal(AppConfig)

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("DiarizeFlow - 即時字幕與音訊設定")
        self.resize(520, 520)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #e2e8f0;
                font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", "Segoe UI Emoji", system-ui, sans-serif;
            }
            QLabel {
                color: #94a3b8;
                font-size: 13px;
                font-weight: 500;
                padding: 2px 0;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                min-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #f8fafc;
                selection-background-color: #38bdf8;
                selection-color: #0f172a;
                padding: 4px;
            }
            QPushButton {
                background-color: #38bdf8;
                color: #0f172a;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #7dd3fc;
            }
            QCheckBox {
                color: #e2e8f0;
                font-size: 13px;
                spacing: 8px;
                padding: 2px 0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Hardware Calibration Status Card
        calib_file = resolve_app_path("models/.calibrated")
        is_calibrated = getattr(self.config, "hardware_calibrated", False) or calib_file.exists()
        diar_prec = "FP16 (Tensor Core)" if getattr(self.config.diarization, "use_fp16", True) else "INT8/FP32"
        status_box = QLabel(
            f"⚡ <b>硬體加速與量化狀態</b>: "
            f"{'✓ 已完成硬體適配與量化校準 (秒開模式, ' + diar_prec + ')' if is_calibrated else '⚙️ 尚未執行自適應量化'}"
        )
        status_box.setStyleSheet("""
            QLabel {
                background-color: rgba(16, 185, 129, 0.12);
                color: #34d399;
                border: 1px solid rgba(16, 185, 129, 0.35);
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
            }
        """)
        layout.addWidget(status_box)

        form = QFormLayout()
        form.setSpacing(12)

        # 1. Microphone device
        self.mic_combo = QComboBox()
        self.loopback_combo = QComboBox()
        self._populate_audio_devices()
        form.addRow("麥克風裝置:", self.mic_combo)
        form.addRow("系統音訊 (遊戲/瀏覽器):", self.loopback_combo)

        # Audio gain control
        self.agc_check = QCheckBox("啟用輸入音量智慧調節 (AGC)")
        self.agc_check.setChecked(getattr(self.config.audio, "agc_enabled", True))
        self.agc_check.setToolTip("自動放大過小聲音、壓限過大聲音，確保 ASR 語音特徵處於最佳動態範圍")
        form.addRow("音量動態調節:", self.agc_check)

        # Dual-track audio routing mode
        self.routing_combo = QComboBox()
        self.routing_combo.addItem("智慧活動動態分流 (抗迴音/無失真, 推薦)", "smart")
        self.routing_combo.addItem("傳統相加混音 (Legacy Additive Mix)", "mix")
        self.routing_combo.addItem("僅收本機麥克風", "mic_only")
        self.routing_combo.addItem("僅收電腦系統聲音", "loopback_only")
        cur_routing = getattr(self.config.audio, "routing_mode", "smart")
        idx = self.routing_combo.findData(cur_routing)
        if idx >= 0:
            self.routing_combo.setCurrentIndex(idx)
        form.addRow("雙軌音訊路由模式:", self.routing_combo)

        # 2. ASR Engine & Faster-Whisper Configuration
        self.asr_combo = QComboBox()
        self.asr_combo.addItem("SenseVoiceSmall (超低延遲 60~80ms)", "sensevoice")
        self.asr_combo.addItem("Faster-Whisper (高精準度 150~250ms)", "faster-whisper")
        cur_engine = getattr(self.config.asr, "engine", "sensevoice")
        self.asr_combo.setCurrentIndex(1 if "whisper" in cur_engine.lower() else 0)
        form.addRow("語音辨識 ASR 引擎:", self.asr_combo)

        self.whisper_model_edit = QLineEdit(getattr(self.config.asr, "whisper_model", "models/faster-whisper-large-v2"))
        form.addRow("Whisper 模型路徑:", self.whisper_model_edit)

        self.prec_combo = QComboBox()
        self.prec_combo.addItems(["float16", "int8", "float32", "fp8", "w4a16", "nvfp4", "mxfp4"])
        cur_prec = getattr(self.config.asr, "whisper_precision", "float16")
        self.prec_combo.setCurrentText(cur_prec)
        form.addRow("Whisper 量化精度:", self.prec_combo)

        # 3. Target language
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["繁體中文", "English", "日本語", "한국어", "簡體中文", "Français", "Deutsch", "Español"])
        self.lang_combo.setCurrentText(self.config.llm.target_language)
        form.addRow("翻譯目標語言:", self.lang_combo)

        # 4. LLM Provider
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["vllm", "llama.cpp", "openai", "claude", "ollama", "bypass"])
        self.provider_combo.setCurrentText(self.config.llm.provider)
        form.addRow("LLM 翻譯引擎:", self.provider_combo)

        # 5. LLM Base URL
        self.url_edit = QLineEdit(self.config.llm.base_url)
        form.addRow("API 端點 (Base URL):", self.url_edit)

        # 6. LLM Model Name
        self.model_edit = QLineEdit(self.config.llm.model_name)
        form.addRow("模型名稱 (Model):", self.model_edit)

        # 7. Auto-dismiss delay
        self.fade_spin = QDoubleSpinBox()
        self.fade_spin.setRange(1.0, 30.0)
        self.fade_spin.setSingleStep(0.5)
        self.fade_spin.setValue(self.config.ui.fade_out_seconds)
        self.fade_spin.setSuffix(" 秒後淡出消失")
        form.addRow("字幕保留時間:", self.fade_spin)

        # 8. Font size
        self.font_spin = QSpinBox()
        self.font_spin.setRange(14, 48)
        self.font_spin.setValue(self.config.ui.font_size)
        form.addRow("字幕字型大小:", self.font_spin)

        # 8b. Max cards queue limit
        self.max_cards_spin = QSpinBox()
        self.max_cards_spin.setRange(1, 10)
        self.max_cards_spin.setValue(getattr(self.config.ui, "max_cards", 3))
        self.max_cards_spin.setSuffix(" 則對話卡片")
        self.max_cards_spin.setToolTip("多卡片佇列上限：同時並存的講者發言卡片數量 (建議 2 ~ 4 則)")
        form.addRow("多卡片佇列上限:", self.max_cards_spin)

        # 9. Speaker Separation Mode & Threshold
        self.streaming_mode_combo = QComboBox()
        self.streaming_mode_combo.addItem("Low Latency (官方預設 1.04s, 換句即分)", "low_latency")
        self.streaming_mode_combo.addItem("Very Low Latency (超低延遲 0.64s)", "very_low_latency")
        self.streaming_mode_combo.addItem("Ultra-Low Latency (極限延遲 0.32s)", "ultra_low_latency")
        self.streaming_mode_combo.addItem("Offline (批次基準 30.4s)", "offline")
        cur_mode = getattr(self.config.diarization, "streaming_mode", "low_latency")
        idx = self.streaming_mode_combo.findData(cur_mode)
        if idx >= 0:
            self.streaming_mode_combo.setCurrentIndex(idx)
        else:
            self.streaming_mode_combo.setCurrentIndex(0)
        self.streaming_mode_combo.setToolTip("NVIDIA Nemotron-3 官方串流架構延遲設定：以 1.04 秒小區間滑動分析，多人對話輪替時精準分離講者")
        form.addRow("Nemotron 串流分離延遲:", self.streaming_mode_combo)

        self.spk_thresh_spin = QDoubleSpinBox()
        self.spk_thresh_spin.setRange(0.50, 0.98)
        self.spk_thresh_spin.setSingleStep(0.02)
        cur_thresh = getattr(self.config.diarization, "speaker_threshold", 0.82)
        self.spk_thresh_spin.setValue(cur_thresh if cur_thresh >= 0.5 else 0.82)
        self.spk_thresh_spin.setToolTip("聲線相似度門檻（越高越容易判定為新講者，建議 0.78 ~ 0.86）")
        form.addRow("語者分離敏感度:", self.spk_thresh_spin)

        # 10. VAD Voice Activity Sensitivity
        self.vad_thresh_spin = QDoubleSpinBox()
        self.vad_thresh_spin.setRange(0.001, 0.15)
        self.vad_thresh_spin.setSingleStep(0.002)
        self.vad_thresh_spin.setDecimals(4)
        self.vad_thresh_spin.setValue(self.config.vad.energy_threshold)
        self.vad_thresh_spin.setToolTip("語音活動檢測門檻（數值越小越靈敏，微弱聲音如 0.006 也能快速捕捉，建議 0.004 ~ 0.015）")
        form.addRow("VAD 人聲檢測門檻:", self.vad_thresh_spin)

        # 11. Max Speech Accumulation Cap (Duration)
        self.max_speech_spin = QDoubleSpinBox()
        self.max_speech_spin.setRange(2.0, 10.0)
        self.max_speech_spin.setSingleStep(0.5)
        self.max_speech_spin.setDecimals(1)
        self.max_speech_spin.setSuffix(" 秒")
        cur_max_s = getattr(self.config.vad, "max_speech_s", 3.5)
        self.max_speech_spin.setValue(cur_max_s if cur_max_s <= 10.0 else 3.5)
        self.max_speech_spin.setToolTip("單句最大音訊累積上限（長話持續不間斷時的最長切片時間，建議 3.0 ~ 4.5 秒，越短字幕出現越快）")
        form.addRow("單句音訊累積上限:", self.max_speech_spin)

        # 12. Silence Timeout
        self.silence_timeout_spin = QSpinBox()
        self.silence_timeout_spin.setRange(150, 800)
        self.silence_timeout_spin.setSingleStep(50)
        self.silence_timeout_spin.setSuffix(" 毫秒 (ms)")
        cur_silence = getattr(self.config.vad, "silence_timeout_ms", 300)
        self.silence_timeout_spin.setValue(cur_silence)
        self.silence_timeout_spin.setToolTip("靜音截斷換句間隔（一句話講完後停頓多少時間即判定句子結束，建議 250 ~ 350 ms）")
        form.addRow("斷句停頓間隔 (Silence):", self.silence_timeout_spin)

        # 13. UI Opacity (Transparency)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.15, 0.95)
        self.opacity_spin.setSingleStep(0.05)
        cur_op = getattr(self.config.ui, "opacity", 0.55)
        self.opacity_spin.setValue(cur_op if cur_op <= 0.85 else 0.55)
        self.opacity_spin.setToolTip("視窗半透明度（數值越小越透明，建議 0.35 ~ 0.60）")
        form.addRow("視窗透明度 (Opacity):", self.opacity_spin)

        # 14. Show original text
        self.show_orig_check = QCheckBox("同時顯示原始辨識語音")
        self.show_orig_check.setChecked(self.config.ui.show_original)
        form.addRow("", self.show_orig_check)

        # 15. View & Clear Logs
        log_box = QHBoxLayout()
        log_btn = QPushButton("📋 打開日誌")
        log_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        log_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #38bdf8;
                font-weight: 600;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #0284c7; color: #ffffff; }
        """)
        log_btn.clicked.connect(self._open_log_file)
        log_box.addWidget(log_btn)

        clear_log_btn = QPushButton("🗑️ 清空日誌")
        clear_log_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #f87171;
                border: 1px solid #ef4444;
                font-weight: 600;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #b91c1c; color: #ffffff; }
        """)
        clear_log_btn.clicked.connect(self._clear_log_file)
        log_box.addWidget(clear_log_btn)

        form.addRow("系統運作日誌:", log_box)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("background-color: #334155; color: #f8fafc;")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("儲存設定")
        save_btn.clicked.connect(self._save_and_apply)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _open_log_file(self):
        if getattr(sys, "frozen", False):
            log_path = Path(sys.executable).parent / "diarizeflow.log"
        else:
            log_path = Path(__file__).resolve().parent.parent.parent.parent / "diarizeflow.log"
        if not log_path.exists():
            try:
                log_path.touch()
            except Exception:
                pass
        try:
            if sys.platform == "win32":
                os.startfile(str(log_path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as e:
            print(f"[!] 無法打開日誌: {e}")

    def _clear_log_file(self):
        if getattr(sys, "frozen", False):
            log_path = Path(sys.executable).parent / "diarizeflow.log"
        else:
            log_path = Path(__file__).resolve().parent.parent.parent.parent / "diarizeflow.log"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 日誌已由使用者清空\n")
            QMessageBox.information(self, "成功", "已成功清空 diarizeflow.log 日誌！")
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"清空日誌失敗: {e}")

    def _populate_audio_devices(self):
        mics, loopbacks = list_audio_devices()
        self.mic_combo.addItem("【未選取/關閉麥克風】", -1)
        for mic in mics:
            self.mic_combo.addItem(mic.display_name(), mic.id)

        self.loopback_combo.addItem("【未選取/關閉系統聲音】", -1)
        for loop in loopbacks:
            self.loopback_combo.addItem(loop.display_name(), loop.id)

        # Set current selection
        mic_dev = self.config.audio.mic_device
        if mic_dev is not None and mic_dev != -1 and mic_dev != "-1":
            idx = self.mic_combo.findData(mic_dev)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)
        else:
            self.mic_combo.setCurrentIndex(0)

        loop_dev = self.config.audio.loopback_device
        if loop_dev is not None and loop_dev != -1 and loop_dev != "-1":
            idx = self.loopback_combo.findData(loop_dev)
            if idx >= 0:
                self.loopback_combo.setCurrentIndex(idx)
            else:
                for i in range(self.loopback_combo.count()):
                    if str(loop_dev) in self.loopback_combo.itemText(i):
                        self.loopback_combo.setCurrentIndex(i)
                        break
        elif loopbacks:
            # Default to first loopback if available
            self.loopback_combo.setCurrentIndex(1)

    def _save_and_apply(self):
        self.config.audio.mic_device = self.mic_combo.currentData()
        self.config.audio.loopback_device = self.loopback_combo.currentData()
        self.config.audio.agc_enabled = self.agc_check.isChecked()
        self.config.audio.routing_mode = self.routing_combo.currentData()
        self.config.asr.engine = self.asr_combo.currentData()
        self.config.asr.whisper_model = self.whisper_model_edit.text().strip()
        self.config.asr.whisper_precision = self.prec_combo.currentText()
        self.config.llm.target_language = self.lang_combo.currentText()
        self.config.llm.provider = self.provider_combo.currentText()
        self.config.llm.base_url = self.url_edit.text().strip()
        self.config.llm.model_name = self.model_edit.text().strip()
        self.config.ui.fade_out_seconds = float(self.fade_spin.value())
        self.config.ui.font_size = int(self.font_spin.value())
        self.config.ui.max_cards = int(self.max_cards_spin.value())
        self.config.ui.opacity = float(self.opacity_spin.value())
        self.config.ui.show_original = self.show_orig_check.isChecked()
        self.config.diarization.streaming_mode = self.streaming_mode_combo.currentData()
        self.config.diarization.speaker_threshold = float(self.spk_thresh_spin.value())
        self.config.vad.energy_threshold = float(self.vad_thresh_spin.value())
        self.config.vad.max_speech_s = float(self.max_speech_spin.value())
        self.config.vad.silence_timeout_ms = int(self.silence_timeout_spin.value())

        self.config.save()
        self.settings_saved.emit(self.config)
        self.accept()


__all__ = ["SettingsDialog"]
