# Hold-to-talk for qt-app (GlobalShortcuts portal)

*Status: planned, not implemented. Written 2026-10-08 and archived for later. Re-check the environment facts below before starting, because portal versions move.*. This file was written by `Anthropic Claude Opus 5.5`, as the plan for how we could add a 'press-to-talk' feature. 

## Context

Today, a checked **Mic** box means every utterance the silence detector catches gets typed into the focused app. This plan adds an optional **hold-to-talk** mode:
- Speech is captured only while a global keyboard shortcut is held, and the Mic box must still be checked.
- With the mode off (the default), the app behaves exactly as it does today.

### Decisions already made

- **Transcribe on release.** This is classic push-to-talk: audio is buffered while the shortcut is held, and whisper runs when it is let go. The tuned RMS silence-detection state machine in `audio.py` is **bypassed, not modified**. Typing therefore starts after release, so held modifiers can't combine with the injected keystrokes.
- **Off by default.**
- **Use the official GlobalShortcuts portal (`org.freedesktop.portal.GlobalShortcuts`), not evdev.** The user picks the binding in **GNOME's own dialog**, so the Settings dialog gets a **"Hold to talk" checkbox**, not a key-name text field.

### Why the portal and not evdev

The alternative was to read `/dev/input/event*` directly, as `doubao-say-main/src/doubao_input/trigger/evdev_ptt.py` does.

| | evdev | GlobalShortcuts portal (chosen) |
|---|---|---|
| Key you hold | Any single key, including a bare RIGHTCTRL | GNOME doesn't bind a lone modifier, so it's a combo or a spare non-modifier key |
| How the key is chosen | Our own Settings field | GNOME's dialog; the app can only suggest one |
| One-time setup | `sudo usermod -aG input $USER`, then log out and back in | Approve one GNOME prompt |
| Security | Any program the user runs can read every keystroke | Nothing new granted; the app only hears its own shortcut |
| Does the key reach the focused app? | Yes. A left-Alt tap opens menu bars in Firefox and LibreOffice | No, GNOME consumes it |
| Works on | X11 and every Wayland desktop | GNOME 48+, KDE, Hyprland. Not Sway; patchy on X11 |

If single-key RIGHTCTRL ever becomes more important than the security and cleanliness of the portal, evdev is the fallback. The listener in this plan is deliberately just `pressed` / `released` signals, so an evdev backend could be added later without touching the recorder or the window.

### Environment facts at the time of writing

These were checked on the development machine:
- GNOME Shell 50.1 on Wayland, with xdg-desktop-portal 1.21.1 and xdg-desktop-portal-gnome 50.0.
- The GlobalShortcuts portal is present at **version 1**, with `CreateSession`, `BindShortcuts`, `ListShortcuts` and the `Activated` / `Deactivated` signals. `ConfigureShortcuts` is introspectable, but it belongs to version 2, so don't rely on it.
- `org.freedesktop.host.portal.Registry.Register(s app_id, a{sv} options)` is available. This is how an unsandboxed app declares its app id.

Re-check them with:

```bash
busctl --user get-property org.freedesktop.portal.Desktop /org/freedesktop/portal/desktop org.freedesktop.portal.GlobalShortcuts version
```

## Step 0: Spike, before any app code

Write a throwaway script, outside the repo, run with `uv run python spike.py` from `qt-app/`. It should answer the unknowns this design rests on. **If `Deactivated` turns out to be unreliable, stop and reconsider before building anything**, because without it there is no hold-to-talk.

1. **Is an app id needed?** Check whether `CreateSession` and `BindShortcuts` work from an unsandboxed process:
   - without registering anything, and
   - after `Registry.Register("com.lingo.voicetyper", {})` (`APP_ID` from `voice_typer/__init__.py`) on a **dedicated** connection: `QDBusConnection.connectToBus(QDBusConnection.BusType.SessionBus, "lingo-shortcuts")`.
2. **Can PyQt6 marshal `a(sa{sv})`?** That is BindShortcuts' argument type. It probably needs a hand-built `QDBusArgument` using `beginArray` / `beginStructure` / `beginMap`. Record the working recipe, the same way `keyboard.py`'s docstring records its four QtDBus facts.
3. **What does GNOME's dialog accept?** Find out whether a single non-modifier key (Pause, Insert, an F-key) can be bound, or only combos. Bind with **no `preferred_trigger`**, so the user chooses and the app doesn't steal something like Ctrl+Space from VS Code.
4. **Does `Deactivated` fire promptly on release?** Also check:
   - whether holding the key produces repeated `Activated` events from autorepeat, and
   - what happens when the modifier is released before the main key of a combo.
5. **Does the binding persist?** After a restart, does `BindShortcuts` with the same id return the stored trigger without prompting again?
6. **How do you rebind or revoke the shortcut in GNOME 50?** Find the place in GNOME Settings, and check whether a v1 portal returns an error for `ConfigureShortcuts`. The answer goes into the README and the help text.

## Implementation

### 1. New `voice_typer/portal.py`: shared request/response plumbing

Move the generic portal plumbing out of `KeyboardInjector` (`voice_typer/keyboard.py`) into a `PortalClient(QObject)` base class:
- `_generate_token`
- `_get_request_path`
- `_subscribe` / `_unsubscribe`
- `_call_async`, which now takes the interface name and calls an overridable `_fail()`

The base class takes a `QDBusConnection`. `KeyboardInjector` inherits it and keeps using `sessionBus()`, with no behavior change. Nothing in `_send_key`, which must stay blocking, or in the restore-token flow moves.

### 2. New `voice_typer/shortcuts.py`: `HoldToTalkShortcut(PortalClient)`

- **Signals:**
  - `pressed`
  - `released`
  - `ready(str)`, carrying the portal's `trigger_description`, e.g. "Ctrl+Space"
  - `failed(str)`
- **A dedicated D-Bus connection.** Registering an app id there must not change the identity under which the RemoteDesktop session and its saved `portal_restore_token` were granted. Re-granting would force the Remote Desktop permission dialog again, which AGENTS.md explicitly says to avoid.
- **Flow:**
  1. `Registry.Register`, if the spike says it is needed.
  2. `CreateSession`.
  3. `BindShortcuts([("hold-to-talk", {"description": "Hold to talk (Lingo)"})])`.
  4. Emit the trigger description from the response through `ready`.
- **Events:**
  - Subscribe, with `@pyqtSlot` methods, to `Activated` and `Deactivated` on the GlobalShortcuts interface.
  - Filter on our session handle and the shortcut id.
  - Ignore a repeated `Activated` while already held.
  - Expose `is_held()`.
- **Teardown:** `close()` closes the portal Session.
- **Lifetime:** created on the first Mic-on with hold-to-talk enabled; closed when the setting is turned off or the app quits. QtDBus delivers signals on the main thread, so no thread bridge is needed.

### 3. `voice_typer/audio.py`: a separate push-to-talk path

The state machine stays untouched.

- **New state:**
  - `push_to_talk`
  - `_key_held`
  - a pre-roll deque of about 0.3s of chunks, so the first syllable isn't lost when speech starts at the same moment as the press
- **One early branch** at the top of `_audio_callback`: `if self.push_to_talk: self._ptt_callback(...); return`. The existing RMS branch structure and constants stay exactly as they are.
- **`_ptt_callback`:**
  - When the key isn't held, chunks only feed the pre-roll.
  - When it is held, chunks go into `audio_buffer`, and `voiced_frames` and `peak_rms` are counted against `silence_threshold`.
  - The voiced count is used only to reject silent presses, so whisper never receives empty audio and hallucinates text.
- **`set_push_to_talk(enabled)`** resets the per-utterance state.
- **`key_pressed()`** seeds the buffer from the pre-roll and moves the phase to `"speech-detected"`, so the window turns green while the shortcut is held.
- **`key_released()`:** if the duration is at least `MIN_AUDIO_DURATION_S` and the voiced duration is at least `MIN_VOICED_DURATION_S`, start the existing `_process_audio` thread (reused unchanged). Otherwise reset and go back to `"idle"`.

### 4. `voice_typer/config.py`

- Add `"hold_to_talk": False` to `DEFAULT_CONFIG`, and coerce it to `bool` in `load_config` so a hand-edited value can't crash startup.

### 5. `voice_typer/settings_dialog.py`

- Add a `hold_to_talk_changed = pyqtSignal(bool)` signal and a `hold_to_talk` constructor argument.
- Add a new "Hold to talk" `QCheckBox`, plus a muted "Shortcut: …" label:
  - The window updates the label through `set_shortcut_description(text)`.
  - It shows "not bound yet" until the portal answers.
- Add a `HOLD_TO_TALK_HELP` paragraph covering:
  - Hold the shortcut while speaking; the text is typed on release.
  - GNOME asks you to choose the shortcut the first time.
  - Where to rebind it (from spike step 6).
- The dialog still never touches the config file or the recorder. It emits, and the window applies.

### 6. `voice_typer/window.py`

- **Stylesheet scoping.** The dialog will now contain a QCheckBox. A stylesheet set on a widget also applies to child dialogs, so the window's `QCheckBox` rules would leak into it: the large font, and black text during the white "typing" phase. The `build_stylesheet` docstring currently relies on the dialog having no QCheckBox. To fix this:
  - Give the mic checkbox `setObjectName("micCheckbox")`.
  - Scope all three checkbox rules to `QCheckBox#micCheckbox`.
  - Update that docstring.
  - The `apply_checkboxes` call and the phase repolish of the mic checkbox are unaffected.
- **Settings wiring.** `open_settings` passes `hold_to_talk` and connects `hold_to_talk_changed` to `_update_hold_to_talk`, which saves the config and calls `_apply_hold_to_talk()` if recording.
- **`_apply_hold_to_talk()`** runs at the end of `start_recording` and after a settings change:
  - **Off:** call `recorder.set_push_to_talk(False)` and close any shortcut session.
  - **On:** create `HoldToTalkShortcut` if it doesn't exist, connect `pressed` / `released` to the recorder's `key_pressed` / `key_released`, and call `set_push_to_talk(True)`.
  - `ready` updates the dialog's label and prints `🎙️ Hold-to-talk: <trigger>`.
  - `failed` shows a `QMessageBox.warning` and unchecks the Mic. It deliberately doesn't fall back to an open mic, which would type things the user didn't expect.
- **Auto-off.** `_on_trigger_pressed` also resets `_last_audio_detection_time`, so holding the shortcut counts as activity.
- **Deferred typing.** In `on_speech_transcribed`, if the shortcut is held because the next utterance has already started, queue the text and type it on release. Otherwise the held modifiers would turn the injected keystrokes into shortcuts.
- **Teardown.** `closeEvent` closes the shortcut session.

### 7. Docs

- **`qt-app/README.md`:**
  - A new "Hold-to-talk" section covering:
    - How it works: hold, speak, release, and the window colors.
    - The first-run GNOME dialog.
    - How to rebind or revoke the shortcut.
    - It needs GNOME 48+ or KDE.
    - Don't pick a combo an app you use relies on, because GNOME consumes it globally.
  - A note in "How It Works".
  - Rows for `portal.py` and `shortcuts.py` in the Code Layout table.
- **`AGENTS.md`**, under "Deliberate non-obvious choices", new entries for:
  - The dedicated D-Bus connection, and why.
  - The `a(sa{sv})` marshalling recipe from the spike.
  - Push-to-talk bypasses the RMS state machine through one early branch.
  - Typing is deferred while the shortcut is held.
  - The `#micCheckbox` stylesheet scoping.
- **`AGENTS.md`'s "Config file" bullet** also gains `hold_to_talk`.

## Verification

1. The spike results are recorded, and the go/no-go on `Deactivated` passed.
2. Run `./run.sh` with hold-to-talk off. It behaves exactly as today: silence-triggered, with no portal session created (check `voice_typer.log`).
3. Enable hold-to-talk in Settings with the Mic on. GNOME's dialog appears, a shortcut is chosen, and the label shows it. Then:
   - Speaking without holding the shortcut does nothing.
   - Holding it turns the window green.
   - Releasing it goes orange, then white, and the text is typed into, for example, gedit.
   - A silent press and release goes back to idle with no whisper call.
4. Hold the shortcut while VS Code or Firefox is focused: the key does not reach the app.
5. Restart the app. There is no GNOME prompt and no Remote Desktop prompt, so the existing `portal_restore_token` still works. `hold_to_talk: true` has persisted in `~/.config/lingo-gtk.yaml`.
6. Press the shortcut again while the previous text is still being typed. That text waits until release and isn't mangled into shortcuts.
7. During the white "typing" phase, the Settings dialog's text stays readable, which confirms the stylesheet scoping.
