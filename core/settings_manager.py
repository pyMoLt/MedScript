# core/settings_manager.py — Persistent user settings loader and config patcher

# Persistent user settings. Loads ~/.medskript/settings.json
# and patches the config module at runtime via setattr().
#
# IMPORTANT: Must be loaded first after deadlock env vars in main.py,
# so all other modules see the correct values from config.

import json
import importlib
from pathlib import Path

import config


# Singleton-style manager for persistent user settings.
class UserSettings:
    """
    Singleton-style manager for persistent user settings.
    Patches the config module directly with saved values.
    """

    def __init__(self):
        self.settings_file: Path = config.USER_SETTINGS_FILE
        self.data: dict = {}
        self.load()

    # Reads settings.json and patches the config module.
    def load(self) -> None:
        """Reads settings.json and patches the config module."""
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)

        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                raw = {}
        else:
            raw = {}

        for key, value in raw.items():
            if key not in config.PERSISTABLE_KEYS:
                continue
            try:
                value = self._coerce(key, value)
                self.data[key] = value
                setattr(config, key, value)
            except Exception:
                pass  # Ungültige Werte ignorieren

        # Rebuild LM_STUDIO_BASE_URL after host/port change
        self._rebuild_base_url()

    # Writes current state as JSON to disk.
    def save(self) -> None:
        """Writes current state as JSON to disk."""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = {}
            for k, v in self.data.items():
                if isinstance(v, Path):
                    serializable[k] = str(v)
                else:
                    serializable[k] = v
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ settings_manager: Speichern fehlgeschlagen: {e}")

    # Reads value directly from config (single source of truth).
    def get(self, key: str):
        """Reads value directly from config (single source of truth)."""
        return getattr(config, key, None)

    # Sets value in data + patches config. Does NOT auto-save.
    def set(self, key: str, value) -> None:
        """Sets value in data + patches config. Does NOT auto-save."""
        if key not in config.PERSISTABLE_KEYS:
            raise ValueError(f"Schlüssel '{key}' ist nicht persistierbar.")
        value = self._coerce(key, value)
        self.data[key] = value
        setattr(config, key, value)
        if key in ("LM_STUDIO_HOST", "LM_STUDIO_PORT"):
            self._rebuild_base_url()

    # Returns dict of all persistable values, suitable for UI display.
    def get_all_for_ui(self) -> dict:
        """
        Returns dict of all persistable values, suitable for UI display.
        Format: {key: {'value': ..., 'type': str}}
        """
        result = {}
        for key in config.PERSISTABLE_KEYS:
            current = getattr(config, key, None)
            result[key] = {
                "value": str(current) if isinstance(current, Path) else current,
                "type": type(current).__name__,
            }
        return result

    # Deletes settings.json and reloads config defaults via importlib.reload.
    def reset_to_defaults(self) -> None:
        """Deletes settings.json and reloads config defaults via importlib.reload."""
        try:
            if self.settings_file.exists():
                self.settings_file.unlink()
        except Exception:
            pass
        self.data = {}
        importlib.reload(config)

    # ── Internal helpers ───────────────────────────────────────────────────────

    # Converts value to the type of the current config value.
    def _coerce(self, key: str, value):
        """Converts value to the type of the current config value."""
        current = getattr(config, key, None)
        if current is None:
            return value
        target_type = type(current)

        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == Path:
            return Path(value)
        else:
            return str(value)

    # Rebuilds LM_STUDIO_BASE_URL after host/port change.
    def _rebuild_base_url(self) -> None:
        """Rebuilds LM_STUDIO_BASE_URL after host/port change."""
        host = getattr(config, "LM_STUDIO_HOST", "127.0.0.1")
        port = getattr(config, "LM_STUDIO_PORT", "1234")
        setattr(config, "LM_STUDIO_BASE_URL", f"http://{host}:{port}/v1")


# ── Global singleton instance ──────────────────────────────────────────────────

_instance: UserSettings | None = None


# Returns the global UserSettings instance. Creates it on first call.
def get_settings() -> UserSettings:
    """Returns the global UserSettings instance. Creates it on first call."""
    global _instance
    if _instance is None:
        _instance = UserSettings()
    return _instance
