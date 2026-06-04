# 📱 NHAI Biometric Attendance Mobile Client (`@workspace/mobile`)

This folder contains the cross-platform mobile client application built on **Expo**, **React Native**, and **TypeScript**. It is designed to work in disconnected field environments (e.g., highway construction sites), performing secure biometric validation locally on the client and syncing ledger logs when internet connectivity is available.

---

## 📂 Navigation & File Structure

```
├── app/
│   ├── (tabs)/
│   │   ├── _layout.tsx         # Tab router architecture configuration
│   │   └── index.tsx           # Home Dashboard (Shows status, sync counters, employee grid)
│   ├── _layout.tsx             # Main router provider & global layout theme
│   ├── liveness.tsx            # Camera view + Multi-Stage Liveness & Biometric Extractor
│   ├── success.tsx             # Successful check-in/enrollment screen
│   ├── failure.tsx             # Liveness failure / profile mismatch screen
│   └── denied.tsx              # Generic security / permission error screen
├── assets/
│   ├── detector.onnx           # Face Bounding Box Local ML detector model
│   ├── face_model_quant.onnx   # Facial embedding generator model (512-dim)
│   └── moire.onnx              # Screen spoofs/Moiré texture pattern classification model
├── components/
│   ├── ErrorBoundary.tsx       # Catches rendering crashes and prevents app teardowns
│   └── ErrorFallback.tsx       # User-friendly crash representation screen
├── constants/
│   ├── colors.ts               # Layout tokens & styling themes
│   ├── localDb.ts              # SQLite database driver config (Offline ledger tables)
│   ├── localDb.web.ts          # Web fallback mock SQLite driver
│   └── vault.ts                # Biometric cryptography (AES templates & Cosine Sim calculations)
└── hooks/
    └── useColors.ts            # Dynamic layout color retrieval hooks
```

---

## ⚙️ Core Technical Workflows

### 🎥 1. Biometric Scanner & Liveness (`app/liveness.tsx`)
This is the core interface of the app. It controls the camera and coordinates four liveness checks:
1. **Photometric Depth Verification**: Flashes the screen at maximum brightness from four quadrants sequentially (Top, Right, Bottom, Left). It decodes the captured frames and analyzes brightness ratios. Real faces have 3D contours that reflect directional light asymmetrically, whereas flat paper/screen printouts reflect light uniformly.
2. **ONNX Moiré Classifier**: Feeds an expanded cropped face bounding box into `moire.onnx` (a model specialized in texture classification) to flag digital pixel refresh grids indicative of screen presentation attacks.
3. **rPPG Heart Rate Monitor**: Monitors forehead green color channel variance over 8 seconds. It utilizes a bandpass filter (0.7–4 Hz) to verify the presence of human heart rate periodicity.
4. **IMU Match**: Subscribes to the device gyroscope (`expo-sensors`) to correlate physical device tilt with optical flow coordinates of the face box.

### 🧠 2. Local ONNX Machine Learning Inference
If liveness is verified, the app captures a face frame and runs:
1. **Face Detection**: Converts the frame to RGB, resizes it to 320x240, and inputs it to `detector.onnx` using `onnxruntime-react-native`. Runs Non-Maximum Suppression (NMS) to isolate the face box.
2. **Biometric Extraction**: Crops the face box, resizes it to 112x112, normalizes pixels, and inputs it to `face_model_quant.onnx` to generate a 512-dimensional floating-point vector embedding.

### 🔒 3. Biometric Security & SQLite Database
* **Encryption**: Embedding vectors are encrypted in memory (`constants/vault.ts`) before being stored in the local SQLite database.
* **Blockchain-lite Ledger**: Attendance logs are saved to the SQLite database `offline_attendance` table, with each record linked to the previous one via a SHA-256 hash.
* **Privacy Purge**: Upon a successful sync with the API server, the local SQLite database clears the employee biometric templates (`encrypted_embeddings = 'SECURELY_CLEARED'`) to protect privacy, while retaining basic logs to show attendance history locally.

---

## 🛠️ Run & Development Scripts

Run these scripts inside `artifacts/mobile/` or using the monorepo root workspace runner:

* **Start Metro Bundler** (Runs expo package manager):
  ```bash
  pnpm run dev
  ```
* **Run Web Mocker**:
  ```bash
  pnpm run serve
  ```
* **Typecheck components**:
  ```bash
  pnpm run typecheck
  ```
* **Run native builds**:
  ```bash
  pnpm run android
  pnpm run ios
  ```
