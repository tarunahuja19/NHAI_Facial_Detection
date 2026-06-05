# 🛣️ NHAI Biometric Attendance & Liveness Verification System

Welcome to the **National Highways Authority of India (NHAI) Biometric Attendance & Liveness Verification System** monorepo. This system is designed for high-security, offline-first attendance logging across NHAI toll plaza construction sites and regional offices. 

By combining **on-device Machine Learning (ONNX)** with a **multi-stage cryptographic liveness pipeline** and a **tamper-proof blockchain-lite ledger**, this application guarantees that logs are genuine, non-repudiable, and fully verified before sync.

---

## 🧭 Repository Navigation Guide

To help you navigate the codebase, here is the functional dependency map of the workspace:

```mermaid
graph TD
    api-spec[lib/api-spec: OpenAPI Contract] -->|orval codegen| api-client[lib/api-client-react: React Query Hooks]
    api-spec -->|orval codegen| api-zod[lib/api-zod: Zod Schema Validators]
    
    api-zod -->|Validation Rules| api-server[artifacts/api-server: Express Backend]
    api-zod -->|Schema Matching| mobile[artifacts/mobile: Expo Mobile App]
    api-client -->|HTTP Sync Client| mobile
    
    db[lib/db: Drizzle Postgres Schema] -->|Queries & Updates| api-server
    db -->|Admin Queries| scripts[scripts: Database Inspections]
```

### 📂 Directory Tree Layout

```
nhai-app/
├── artifacts/
│   ├── api-server/         # Express API Backend (Postgres, Ledger Verification)
│   └── mobile/             # Expo React Native App (ONNX runtime, SQLite, Liveness)
├── lib/
│   ├── api-client-react/   # React Query Network Hooks
│   ├── api-spec/           # OpenAPI Contracts & Codegen
│   ├── api-zod/            # Zod validation schemas
│   └── db/                 # Drizzle Postgres ORM & Schema
├── model-training/         # ML Pipelines (JAX Recognition & TF/Keras Detection)
├── scripts/                # Database Inspection & Utility CLI Scripts
└── package.json            # Workspace Configuration
```

### 📂 Directory Directory Breakdown

#### 1. Core Applications (`/artifacts`)
* **[`artifacts/mobile`](file:///home/jemin/Desktop/Code/nhai-app/artifacts/mobile)**: The user-facing cross-platform Expo application. It accesses the device camera to register employees, executes on-device facial identification, runs a 4-stage hardware/software liveness pipeline, logs records in a local SQLite file, and schedules network sync.
* **[`artifacts/api-server`](file:///home/jemin/Desktop/Code/nhai-app/artifacts/api-server)**: Node/Express central API backend. Handles delta synchronization, executes employee biometric identity averaging (conflict resolution), and verifies log cryptographic chains before persistence.

#### 2. Shared Libraries & Contracts (`/lib`)
* **[`lib/api-spec`](file:///home/jemin/Desktop/Code/nhai-app/lib/api-spec)**: OpenAPI specification (`openapi.yaml`) mapping network contracts and Orval configurations.
* **[`lib/api-client-react`](file:///home/jemin/Desktop/Code/nhai-app/lib/api-client-react)**: Automated React/React Native networking package exporting custom-configured TanStack queries.
* **[`lib/api-zod`](file:///home/jemin/Desktop/Code/nhai-app/lib/api-zod)**: Shared validation schemas ensuring runtime data compliance across both client and server boundaries.
* **[`lib/db`](file:///home/jemin/Desktop/Code/nhai-app/lib/db)**: Single database controller housing PostgreSQL driver initializers and Drizzle ORM definitions.

#### 3. Machine Learning Pipelines (`/model-training`)
* **[`model-training`](file:///home/jemin/Desktop/Code/nhai-app/model-training)**: Contains the training, evaluation, ONNX export, and post-quantization scripts for the face detection and recognition/liveness models.

#### 4. Administrative Utilities (`/scripts`)
* **[`scripts`](file:///home/jemin/Desktop/Code/nhai-app/scripts)**: Internal developer utility scripts. Includes `view-db.ts` for database table printing and model encryption setups.

---

## 🔍 How the System Works (End-to-End Flow)

Below is the execution flow from the moment an employee checks in to the final server synchronization:

```mermaid
sequenceDiagram
    autonumber
    actor Employee
    participant Mobile as Mobile App (SQLite/ONNX)
    participant Server as Central API Server (Postgres)
    
    Employee->>Mobile: Input Employee ID & Request Attendance
    Note over Mobile: Loads ONNX models (detector, moire, recognizer) into RAM
    
    rect rgb(240, 248, 255)
        Note over Mobile: Multi-Stage Liveness Checks
        Mobile->>Mobile: 1. Photometric Stereo (Flash screen & verify 3D face structure)
        Mobile->>Mobile: 2. ONNX Moiré Check (Validate texture / anti-spoofing)
        Mobile->>Mobile: 3. rPPG Blood Pulse (Assess micro-color variance on forehead)
        Mobile->>Mobile: 4. IMU Correlation (Match gyroscopic motion to face box shift)
    end
    
    Mobile->>Mobile: On-Device Face Recognition (512-dim embedding)
    
    alt Liveness Failed or Cosine Similarity < Threshold
        Mobile-->>Employee: Show Failure Screen
    else Verification Passed
        Note over Mobile: Fetch latest local block hash
        Mobile->>Mobile: Compute record hash: SHA256(prevHash | id | time | embedHash)
        Mobile->>Mobile: Save to Offline SQLite Ledger (is_synced = 0)
        Mobile-->>Employee: Show Success Screen
    end
    
    Note over Mobile: Connection Restored / Click Sync
    Mobile->>Server: POST /api/sync (Payload: Delta Enrollments & Attendance Ledger Chain)
    
    rect rgb(255, 245, 230)
        Note over Server: Server-Side Integrity Check
        Server->>Server: Re-verify full cryptographic chain matching prevHash links
        alt Chain Integrity Broken (Tampering Detected)
            Server-->>Mobile: Reject Sync with 400 Bad Request
        else Chain Valid
            Server->>Server: Merge new biometric templates (Average embeddings on conflict)
            Server->>Server: Insert records to Postgres
            Server-->>Mobile: Respond 200 OK (Acknowledge Synced IDs)
        end
    end
    
    Mobile->>Mobile: Mark attendance as synced. secure-clear local biometric templates!
```

---

## 🛠️ Deep Dive: The Core Systems

### 🛡️ The 4-Stage Liveness Pipeline
Spoofing a standard camera check-in with a photo printout or digital tablet is straightforward. This project implements four overlapping checks to verify presence:
1. **Screen Flash (Photometric Stereo)**: By rapidly flashing the screen white and analyzing brightness variations at the left, right, top, and bottom sectors of the face, the app detects the presence of actual 3D human depth (flat surfaces return uniform reflections).
2. **FFT Moiré Classification**: Captures the face frame, feeds a cropped zone into `moire.onnx`, and detects screen pixel structures.
3. **rPPG Forehead Extraction**: Blood flow causes imperceptible color changes in skin tissue. The camera tracks forehead green-channel intensity changes. A bandpass filter (0.7 Hz to 4.0 Hz) evaluates the periodicity of the signal to identify a human pulse.
4. **IMU motion match**: Correlates the accelerometer and gyroscope data of the physical device with the optical coordinate shift of the face to confirm they move together.

### 🧠 Model Architectures

The system uses two custom deep neural networks optimized for local edge execution:

#### 1. Face Recognition & Liveness Model (`Iris`)
Implemented in **JAX/Flax**, this model is optimized to fit within a strict **4 MB INT8 memory limit** (approx. 4.08M parameters) for deployment on edge devices.

```mermaid
graph TD
    Input["Input: 112x112x3"] --> Stem["Stem (3 Conv Layers)<br/>Output: 28x28x64"]
    Stem --> Stage1["Stage 1 (C=32, Blocks=2)<br/>RepMix + FFNx2<br/>Output: 28x28x32"]
    
    Stage1 --> Down1["Downsample (Stride 2)<br/>Output: 14x14x64"]
    Down1 --> Stage2["Stage 2 (C=64, Blocks=2)<br/>RepMix + LiteMHLA + FFN<br/>Output: 14x14x64"]
    
    Stage2 --> Down2["Downsample (Stride 2)<br/>Output: 7x7x128"]
    Down2 --> Stage3["Stage 3 (C=128, Blocks=12)<br/>RepMix + LiteMHLA + FFN<br/>Output: 7x7x128"]
    
    Stage3 --> Down3["Downsample (Stride 2)<br/>Output: 4x4x256"]
    Down3 --> Stage4["Stage 4 (C=256, Blocks=6)<br/>RepMix + LiteMHLA + LearnedGAT + FFN<br/>Output: 4x4x256"]
    
    Stage1 --> Fusion["Multi-Scale Fusion<br/>Resizes 28x28, 14x14, 7x7 maps<br/>to 4x4 resolution"]
    Stage2 --> Fusion
    Stage3 --> Fusion
    Stage4 --> Fusion
    
    Fusion --> Head["Recognition Head<br/>GAP -> Dense(256) -> GELU -> Dense(512)<br/>Output: 512-dim embedding"]
```

* **RepMix Token Mixer**: A structurally re-parameterizable depthwise mixer. During training, it branches into depthwise-KxK, depthwise-1x1, and identity branches to capture multi-scale spatial details. During inference, these collapse into a single standard depthwise conv, reducing execution latency by ~3x.
* **LiteMHLA (Lite Multi-Head Linear Attention)**: Projects channel attention to spatial dimensions, allowing global contextual modeling at Stage 2, 3, and 4.
* **LearnedGAT (Learned Graph Attention)**: Leverages 11 learned semantic region queries that compete over spatial locations to identify facial feature zones. Uses a half-rank bottleneck (`key_dim = dim // 2`) to save 600K parameters.
* **AdaFace Loss**: Adapts margin sizes based on image quality scores calculated using the pre-normalized embedding norm. Ideal for outdoor construction plaza environments with high lighting variability.

#### 2. Face Detection Model (`Face Detector`)
Implemented in **TensorFlow/Keras**, this model utilizes Inverted Residual (MBConv) blocks to extract facial structures and output boundary details.

```mermaid
graph TD
    Input["Input: 128x128x3"] --> Stem["Stem Conv<br/>Output: 64x64x16"]
    
    Stem --> Stage1["Stage 1 MBConv Block<br/>(C=24, Stride=1, Kernel=3)<br/>Output: 64x64x24"]
    Stage1 --> Stage2["Stage 2 MBConv Block<br/>(C=32, Stride=2, Kernel=3)<br/>Output: 32x32x32"]
    Stage2 --> Stage3["Stage 3 MBConv Block<br/>(C=48, Stride=1, Kernel=3)<br/>Output: 32x32x48"]
    Stage3 --> Stage4["Stage 4 MBConv Block<br/>(C=64, Stride=2, Kernel=3)<br/>Output: 16x16x64"]
    Stage4 --> Stage5["Stage 5 MBConv Block<br/>(C=96, Stride=1, Kernel=3)<br/>Output: 16x16x96"]
    Stage5 --> Stage6["Stage 6 MBConv Block<br/>(C=128, Stride=2, Kernel=5)<br/>Output: 8x8x128"]
    Stage6 --> Stage7["Stage 7 MBConv Block<br/>(C=128, Stride=1, Kernel=5)<br/>Output: 8x8x128"]
    
    Stage7 --> GAP["Global Average Pooling<br/>Output: 128 channels"]
    GAP --> Dense["Shared Dense & Dropout<br/>Dense(128) + Dropout(0.3)"]
    
    Dense --> BoxHead["Box Head<br/>Dense(4) -> Sigmoid<br/>Output: Bounding Box [cx,cy,w,h]"]
    Dense --> LandmarkHead["Landmark Head<br/>Dense(10) -> Sigmoid<br/>Output: 5 Coordinates [x,y]"]
    Dense --> ConfHead["Confidence Head<br/>Dense(1) -> Sigmoid<br/>Output: Probability [0,1]"]
```

* **MBConv Blocks**: Features MobileNetV3-style depthwise convolutions, ReLU6 activations, and Squeeze-and-Excitation (SE) channel-attention modules.
* **Task-Specific Head Projections**: Regresses 3 outputs from a shared dense feature vector:
  * **Bounding Box**: Regresses `[cx, cy, w, h]` box parameters.
  * **Landmarks**: Resolves 5 point landmarks (`[x,y]` for eyes, nose, and mouth corners).
  * **Confidence**: Softmax classification for face presence detection.

### ⛓️ Blockchain-Lite Ledger
Each local check-in is logged as a "block" containing:
* `employee_id`
* `timestamp`
* `gps` coordinates
* `embedding_hash` (SHA256 of face recognition output)
* `record_hash` (computed as `SHA256(prev_record_hash | employee_id | timestamp | embedding_hash)`)

If any log row is updated, deleted, or inserted out of order, the chain of hashes breaks. The server checks this chain on every synchronization payload.

---

## 🚦 Quick Start for Developers

### Installation & Workspace Build
1. Clone the repository and install all dependencies:
   ```bash
   pnpm install
   ```
2. Build all core libraries and compile TypeScript project maps:
   ```bash
   pnpm run build
   ```

### Running the Backend Server
1. Move to `lib/db` and create a `.env` file containing your PostgreSQL string:
   ```env
   DATABASE_URL=postgresql://postgres:secretpassword@localhost:5432/nhai_attendance
   ```
2. Push the schema updates:
   ```bash
   pnpm --filter @workspace/db run push
   ```
3. Boot the Express API Server in development mode:
   ```bash
   pnpm --filter @workspace/api-server dev
   ```

### Running the Mobile Client
1. Boot the Expo development server:
   ```bash
   pnpm --filter @workspace/mobile dev
   ```
2. Scan the QR code with your Expo Go app (Android/iOS) or press `w` to launch the client inside your web browser.
