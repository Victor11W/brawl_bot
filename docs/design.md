# Design notes

Detailed notes kept out of the README to keep it concise.

## Platform decision history

Hardware churned during scoping: NVIDIA GPU -> Mac mini -> back to Windows.
Final: Windows + NVIDIA GPU (CUDA). Mac was dropped because LDPlayer 9 is
Windows-only (no native macOS build; only a Windows VM workaround exists).

## Emulator

- LDPlayer 9 on Windows, Brawl Stars installed.
- Fixed window resolution (target 1280x720) so capture and input coordinates
  stay stable.
- ADB must be enabled in LDPlayer settings; connect over `127.0.0.1:5555`.

## Capture

- `mss` grabs the LDPlayer window region into a numpy array (BGR via opencv).
- Capture at a fixed cadence; decision frequency is bounded by capture +
  inference + input latency.

## Detection

- Ultralytics YOLOv11, trained/fine-tuned on the Roboflow dataset
  `bloxxy/brawl-stars-dataset` (10 classes, MIT).
- Classes: Me, Enemy, Friendly, Safe_Enemy, Safe_Friendly, Gem, Ball, PP,
  PP_Box, Hot_Zone.
- The `Safe_*` classes likely mean an entity hidden in a bush (safe), useful
  signal even though we do not detect the bush geometry itself.

## State extraction (to define)

- Convert detections into a structured observation: relative positions of Me vs
  enemies, gems, etc.
- Coordinate normalization relative to frame size.

## RL model (to define)

- Input: full detected state.
- Output: movement direction + boolean shoot action.
- Candidate algorithm: PPO via Stable-Baselines3. Gymnasium environment wrapper.

## Action / input injection

- ADB: swipe to drive the movement joystick, tap to shoot / use super.
- Joystick mapping from a direction vector to swipe coordinates is to be
  calibrated against the LDPlayer resolution.

## Open question: bushes and walls

- YOLO detects mobile entities, not static terrain (bushes, walls).
- Ideal: obtain static map layout data (bush/wall geometry per map) and align it
  to the live frame, instead of detecting terrain from pixels.
- Sources to investigate: community map data, Brawl Stars assets, or manual
  per-map masks. Unresolved.

## Anti-detection / ban risk

- Supercell may flag emulator/automated play. Keep training cautious; consider a
  practice context where possible. Tracked as a risk, not yet mitigated.

## Verification per step

- Step 1 detection:
  - `adb devices` lists the LDPlayer instance.
  - A capture script saves one mss frame of the LDPlayer window.
  - YOLO loads the Roboflow-trained weights and annotates a captured frame.
