# Mobile Client Application (`@workspace/mobile`)

This folder contains the cross-platform Expo React Native mobile application for the NHAI Facial Detection & Attendance system. It is designed to work in remote or offline-first scenarios, utilizing on-device Machine Learning for facial validation and local encryption for data privacy.

---

## 📱 Features & Architecture

### 1. Multi-Stage Biometric Liveness Validation (`liveness.tsx`)
To prevent presentation attacks (photos, screen playbacks, masks), the app executes a multi-phase validation pipeline:
* **Photometric Stereo**: Dynamically changes screen brightness/color from 4 directions (Top, Right, Bottom, Left) and evaluates the captured facial brightness gradients.
* **ONNX Moiré Classifier**: Uses `moire.onnx` to inspect image frequency patterns and texture structures for digital pixels.
* **rPPG Pulse Detection**: Captures green color fluctuations in the forehead region across frames to detect real blood volume pulse cycles.
* **IMU (Inertial Measurement Unit) Correlation**: Tracks gyroscope motion alongside optical flow bounding box movement.

### 2. On-Device ML Pipeline (ONNX Runtime)
* **Face Detection**: Runs `detector.onnx` on resized 320x240 camera frames, filtering boxes via Non-Maximum Suppression (NMS).
* **Face Recognition**: Runs `face_model_quant.onnx` on 112x112 cropped face regions to extract 512-dimensional vector embeddings.
* **Local Embedding Vault**: Biometric templates are encrypted before saving, using key-based obfuscation to protect user identity.

### 3. Offline-First Database Schema (SQLite)
Uses `expo-sqlite` to maintain local records in `nhai_liveness.db`:
* **`enrolled_employees`**: Stores name and encrypted embeddings.
* **`offline_attendance`**: Stores attendance logs linked together via a blockchain-lite hash chain (`record_hash`).
* **Privacy Purge**: Once logs are successfully synced to the central API server, local templates are overwritten with `'SECURELY_CLEARED'` to respect privacy rights, while local attendance logs are flagged as synced (`is_synced = 1`) to preserve user history.

---

## 📂 Project Structure

```
├── app/
│   ├── (tabs)/index.tsx        # Dashboard (List employee details, Sync offline logs)
│   ├── liveness.tsx            # Full-screen camera scanner & liveness verification
│   ├── success.tsx             # Successful check-in result screen
│   ├── failure.tsx             # Failed liveness checks or mismatch screen
│   └── denied.tsx              # Error / access denied screen
├── assets/
│   ├── detector.onnx           # ONNX Face detector model
│   ├── face_model_quant.onnx   # ONNX Face recognizer model
│   └── moire.onnx              # ONNX Screen-spoofing checker model
├── components/                 # Error boundary fallback, UI layout components
├── constants/
│   ├── localDb.ts              # SQLite database manager (Queries & initialization)
│   └── vault.ts                # Cryptographic helper (Cosine similarity & encryption)
└── hooks/                      # Custom hooks (e.g. useColors)
```

---

## 🛠️ Commands

Run these within `artifacts/mobile/` or using the monorepo root workspace runner:

* **Start Expo Bundler**:
  ```bash
  pnpm run dev
  # or
  pnpm run start
  ```
* **Run Web / Serve static backend mocker**:
  ```bash
  pnpm run serve
  ```
* **Run TypeScript Typechecker**:
  ```bash
  pnpm run typecheck
  ```
* **Compile Static Builds**:
  ```bash
  pnpm run build
  ```
* **Run Native Clients (requires SDK setups)**:
  ```bash
  pnpm run android
  pnpm run ios
  ```
