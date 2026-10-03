"""Persistent Voiceprint Profiles and Speaker Database Manager.

Manages persistent speaker voiceprints on disk (e.g. data/speakers/profiles.json).
Supports:
- Zero-shot bootstrapping from enrolled speaker profiles
- Anchor protection & anti-drift assimilation
- Manual speaker pinning and renaming
- Safe atomic JSON serialization
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union
import uuid
import numpy as np

from diarizeflow.app.config import get_data_path, resolve_app_path


@dataclass
class SpeakerProfile:
    """Speaker voiceprint identity and anchor embedding profile."""

    id: str
    name: str
    color: Optional[str] = None
    is_pinned: bool = False
    sample_count: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    anchor_emb: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    last_seen_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        """Convert profile to serializable dictionary format."""
        data: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "is_pinned": self.is_pinned,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
        }
        if include_embedding:
            data["anchor_emb"] = [round(float(x), 6) for x in self.anchor_emb]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerProfile":
        """Instantiate profile from dictionary with embedding normalization."""
        raw_emb = data.get("anchor_emb", [])
        if isinstance(raw_emb, np.ndarray):
            emb = raw_emb.astype(np.float32)
        elif isinstance(raw_emb, list):
            emb = np.array(raw_emb, dtype=np.float32)
        else:
            emb = np.zeros(512, dtype=np.float32)

        norm = float(np.linalg.norm(emb))
        if norm > 1e-6:
            emb /= norm

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return cls(
            id=str(data.get("id", f"spk_{uuid.uuid4().hex[:8]}")),
            name=str(data.get("name", "未知講者")),
            color=data.get("color"),
            is_pinned=bool(data.get("is_pinned", False)),
            sample_count=int(data.get("sample_count", 1)),
            created_at=str(data.get("created_at", now_str)),
            anchor_emb=emb,
            last_seen_at=str(data.get("last_seen_at", now_str)),
        )


class VoiceprintDatabase:
    """Thread-safe persistent database for enrolled and pinned speaker voiceprints."""

    def __init__(
        self,
        storage_path: Union[str, Path] = "data/speakers/profiles.json",
        model_type: str = "nemotron_512d",
    ):
        self.storage_path = Path(storage_path)
        self.model_type = model_type
        self.profiles: Dict[str, SpeakerProfile] = {}
        self._lock = threading.RLock()
        self.load_profiles()

    def get_resolved_path(self) -> Path:
        """Resolve persistent writable path on disk."""
        if self.storage_path.is_absolute():
            return self.storage_path
        return get_data_path(self.storage_path)

    def load_profiles(self) -> List[SpeakerProfile]:
        """Load enrolled profiles from JSON storage."""
        with self._lock:
            path = self.get_resolved_path()
            if not path.exists():
                alt = resolve_app_path(self.storage_path)
                if alt.exists():
                    path = alt
                else:
                    return []

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        return []
                    data = json.loads(content)

                self.model_type = data.get("model_type", self.model_type)
                profile_list = data.get("profiles", [])

                loaded: Dict[str, SpeakerProfile] = {}
                for p_dict in profile_list:
                    if isinstance(p_dict, dict) and "name" in p_dict:
                        prof = SpeakerProfile.from_dict(p_dict)
                        loaded[prof.id] = prof

                self.profiles = loaded
                print(f"[✓] 已從 {path} 載入 {len(self.profiles)} 個永久聲紋講者檔案")
                return list(self.profiles.values())
            except Exception as e:
                print(f"[!] 載入聲紋資料庫失敗 ({path}): {e}")
                return []

    def save_profiles(self) -> None:
        """Persist pinned profiles to disk atomically."""
        with self._lock:
            path = self.get_resolved_path()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Store all pinned profiles
                pinned = [p.to_dict(include_embedding=True) for p in self.profiles.values() if p.is_pinned]
                data = {
                    "version": 1,
                    "model_type": self.model_type,
                    "profiles": pinned,
                }
                tmp_path = path.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                tmp_path.replace(path)
            except Exception as e:
                print(f"[!] 儲存聲紋資料庫至 {path} 失敗: {e}")

    def get_pinned_profiles(self) -> List[SpeakerProfile]:
        """Return list of pinned permanent speaker profiles."""
        with self._lock:
            return [p for p in self.profiles.values() if p.is_pinned]

    def get_all_profiles(self) -> List[SpeakerProfile]:
        """Return all profiles in the database."""
        with self._lock:
            return list(self.profiles.values())

    def find_profile(self, id_or_name: str) -> Optional[SpeakerProfile]:
        """Find profile by exact ID or display name."""
        with self._lock:
            if id_or_name in self.profiles:
                return self.profiles[id_or_name]
            for p in self.profiles.values():
                if p.name == id_or_name:
                    return p
            return None

    def add_or_update(self, profile: SpeakerProfile) -> None:
        """Add or update a speaker profile and persist if pinned."""
        with self._lock:
            self.profiles[profile.id] = profile
            if profile.is_pinned:
                self.save_profiles()

    def delete(self, id_or_name: str) -> bool:
        """Delete a profile by ID or name."""
        with self._lock:
            prof = self.find_profile(id_or_name)
            if prof:
                del self.profiles[prof.id]
                self.save_profiles()
                return True
            return False

    def clear(self) -> None:
        """Clear all in-memory profiles."""
        with self._lock:
            self.profiles.clear()
